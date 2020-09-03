#!/usr/bin/env julia
# -*- julia -*-

# remez.jl - implementation of the Remez algorithm for polynomial approximation
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Base.\

# ----------------------------------------------------------------------
# Helper functions to cope with different Julia versions.
if VERSION >= v"0.7.0"
    array1d(T, d) = Array{T, 1}(undef, d)
    array2d(T, d1, d2) = Array{T, 2}(undef, d1, d2)
else
    array1d(T, d) = Array(T, d)
    array2d(T, d1, d2) = Array(T, d1, d2)
end
if VERSION < v"0.5.0"
    String = ASCIIString
end
if VERSION >= v"0.6.0"
    # Use Base.invokelatest to run functions made using eval(), to
    # avoid "world age" error
    run(f, x...) = Base.invokelatest(f, x...)
else
    # Prior to 0.6.0, invokelatest doesn't exist (but fortunately the
    # world age problem also doesn't seem to exist)
    run(f, x...) = f(x...)
end

# ----------------------------------------------------------------------
# Global variables configured by command-line options.
floatsuffix = "" # adjusted by --floatsuffix
xvarname = "x" # adjusted by --variable
epsbits = 256 # adjusted by --bits
debug_facilities = Set() # adjusted by --debug
full_output = false # adjusted by --full
array_format = false # adjusted by --array
preliminary_commands = array1d(String, 0) # adjusted by --pre

# ----------------------------------------------------------------------
# Diagnostic and utility functions.

# Enable debugging printouts from a particular subpart of this
# program.
#
# Arguments:
#    facility   Name of the facility to debug. For a list of facility names,
#               look through the code for calls to debug().
#
# Return value is a BigFloat.
function enable_debug(facility)
    push!(debug_facilities, facility)
end

# Print a diagnostic.
#
# Arguments:
#    facility   Name of the facility for which this is a debug message.
#    printargs  Arguments to println() if debugging of that facility is
#               enabled.
macro debug(facility, printargs...)
    printit = quote
        print("[", $facility, "] ")
    end
    for arg in printargs
        printit = quote
            $printit
            print($(esc(arg)))
        end
    end
    return quote
        if $facility in debug_facilities
            $printit
            println()
        end
    end
end

# Evaluate a polynomial.

# Arguments:
#    coeffs   Array of BigFloats giving the coefficients of the polynomial.
#             Starts with the constant term, i.e. coeffs[i] is the
#             coefficient of x^(i-1) (because Julia arrays are 1-based).
#    x        Point at which to evaluate the polynomial.
#
# Return value is a BigFloat.
function poly_eval(coeffs::Array{BigFloat}, x::BigFloat)
    n = length(coeffs)
    if n == 0
        return BigFloat(0)
    elseif n == 1
        return coeffs[1]
    else
        return coeffs[1] + x * poly_eval(coeffs[2:n], x)
    end
end

# Evaluate a rational function.

# Arguments:
#    ncoeffs  Array of BigFloats giving the coefficients of the numerator.
#             Starts with the constant term, and 1-based, as above.
#    dcoeffs  Array of BigFloats giving the coefficients of the denominator.
#             Starts with the constant term, and 1-based, as above.
#    x        Point at which to evaluate the function.
#
# Return value is a BigFloat.
function ratfn_eval(ncoeffs::Array{BigFloat}, dcoeffs::Array{BigFloat},
                    x::BigFloat)
    return poly_eval(ncoeffs, x) / poly_eval(dcoeffs, x)
end

# Format a BigFloat into an appropriate output format.
# Arguments:
#    x        BigFloat to format.
#
# Return value is a string.
function float_to_str(x)
    return string(x) * floatsuffix
end

# Format a polynomial into an arithmetic expression, for pasting into
# other tools such as gnuplot.

# Arguments:
#    coeffs   Array of BigFloats giving the coefficients of the polynomial.
#             Starts with the constant term, and 1-based, as above.
#
# Return value is a string.
function poly_to_string(coeffs::Array{BigFloat})
    n = length(coeffs)
    if n == 0
        return "0"
    elseif n == 1
        return float_to_str(coeffs[1])
    else
        return string(float_to_str(coeffs[1]), "+", xvarname, "*(",
                      poly_to_string(coeffs[2:n]), ")")
    end
end

# Format a rational function into a string.

# Arguments:
#    ncoeffs  Array of BigFloats giving the coefficients of the numerator.
#             Starts with the constant term, and 1-based, as above.
#    dcoeffs  Array of BigFloats giving the coefficients of the denominator.
#             Starts with the constant term, and 1-based, as above.
#
# Return value is a string.
function ratfn_to_string(ncoeffs::Array{BigFloat}, dcoeffs::Array{BigFloat})
    if length(dcoeffs) == 1 && dcoeffs[1] == 1
        # Special case: if the denominator is just 1, leave it out.
        return poly_to_string(ncoeffs)
    else
        return string("(", poly_to_string(ncoeffs), ")/(",
                      poly_to_string(dcoeffs), ")")
    end
end

# Format a list of x,y pairs into a string.

# Arguments:
#    xys      Array of (x,y) pairs of BigFloats.
#
# Return value is a string.
function format_xylist(xys::Array{Tuple{BigFloat,BigFloat}})
    return ("[\n" *
            join(["  "*string(x)*" -> "*string(y) for (x,y) in xys], "\n") *
            "\n]")
end

# ----------------------------------------------------------------------
# Matrix-equation solver for matrices of BigFloat.
#
# I had hoped that Julia's type-genericity would allow me to solve the
# matrix equation Mx=V by just writing 'M \ V'. Unfortunately, that
# works by translating the inputs into double precision and handing
# off to an optimised library, which misses the point when I have a
# matrix and vector of BigFloat and want my result in _better_ than
# double precision. So I have to implement my own specialisation of
# the \ operator for that case.
#
# Fortunately, the point of using BigFloats is that we have precision
# to burn, so I can do completely naïve Gaussian elimination without
# worrying about instability.

# Arguments:
#    matrix_in    2-dimensional array of BigFloats, representing a matrix M
#                 in row-first order, i.e. matrix_in[r,c] represents the
#                 entry in row r col c.
#    vector_in    1-dimensional array of BigFloats, representing a vector V.
#
# Return value: a 1-dimensional array X of BigFloats, satisfying M X = V.
#
# Expects the input to be an invertible square matrix and a vector of
# the corresponding size, on pain of failing an assertion.
function \(matrix_in :: Array{BigFloat,2},
           vector_in :: Array{BigFloat,1})
    # Copy the inputs, because we'll be mutating them as we go.
    M = copy(matrix_in)
    V = copy(vector_in)

    # Input consistency criteria: matrix is square, and vector has
    # length to match.
    n = length(V)
    @assert(n > 0)
    @assert(size(M) == (n,n))

    @debug("gausselim", "starting, n=", n)

    for i = 1:1:n
        # Straightforward Gaussian elimination: find the largest
        # non-zero entry in column i (and in a row we haven't sorted
        # out already), swap it into row i, scale that row to
        # normalise it to 1, then zero out the rest of the column by
        # subtracting a multiple of that row from each other row.

        @debug("gausselim", "matrix=", repr(M))
        @debug("gausselim", "vector=", repr(V))

        # Find the best pivot.
        bestrow = 0
        bestval = 0
        for j = i:1:n
            if abs(M[j,i]) > bestval
                bestrow = j
                bestval = M[j,i]
            end
        end
        @assert(bestrow > 0) # make sure we did actually find one

        @debug("gausselim", "bestrow=", bestrow)

        # Swap it into row i.
        if bestrow != i
            for k = 1:1:n
                M[bestrow,k],M[i,k] = M[i,k],M[bestrow,k]
            end
            V[bestrow],V[i] = V[i],V[bestrow]
        end

        # Scale that row so that M[i,i] becomes 1.
        divisor = M[i,i]
        for k = 1:1:n
            M[i,k] = M[i,k] / divisor
        end
        V[i] = V[i] / divisor
        @assert(M[i,i] == 1)

        # Zero out all other entries in column i, by subtracting
        # multiples of this row.
        for j = 1:1:n
            if j != i
                factor = M[j,i]
                for k = 1:1:n
                    M[j,k] = M[j,k] - M[i,k] * factor
                end
                V[j] = V[j] - V[i] * factor
                @assert(M[j,i] == 0)
            end
        end
    end

    @debug("gausselim", "matrix=", repr(M))
    @debug("gausselim", "vector=", repr(V))
    @debug("gausselim", "done!")

    # Now we're done: M is the identity matrix, so the equation Mx=V
    # becomes just x=V, i.e. V is already exactly the vector we want
    # to return.
    return V
end

# ----------------------------------------------------------------------
# Least-squares fitting of a rational function to a set of (x,y)
# points.
#
# We use this to get an initial starting point for the Remez
# iteration. Therefore, it doesn't really need to be particularly
# accurate; it only needs to be good enough to wiggle back and forth
# across the target function the right number of times (so as to give
# enough error extrema to start optimising from) and not have any
# poles in the target interval.
#
# Least-squares fitting of a _polynomial_ is actually a sensible thing
# to do, and minimises the rms error. Doing the following trick with a
# rational function P/Q is less sensible, because it cannot be made to
# minimise the error function (P/Q-f)^2 that you actually wanted;
# instead it minimises (P-fQ)^2. But that should be good enough to
# have the properties described above.
#
# Some theory: suppose you're trying to choose a set of parameters a_i
# so as to minimise the sum of squares of some error function E_i.
# Basic calculus says, if you do this in one variable, just
# differentiate and solve for zero. In this case, that works fine even
# with multiple variables, because you _partially_ differentiate with
# respect to each a_i, giving a system of equations, and that system
# turns out to be linear so we just solve it as a matrix.
#
# In this case, our parameters are the coefficients of P and Q; to
# avoid underdetermining the system we'll fix Q's constant term at 1,
# so that our error function (as described above) is
#
# E = \sum (p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d)^2
#
# where the sum is over all (x,y) coordinate pairs. Setting dE/dp_j=0
# (for each j) gives an equation of the form
#
# 0 = \sum 2(p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d) x^j
#
# and setting dE/dq_j=0 gives one of the form
#
# 0 = \sum 2(p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d) y x^j
#
# And both of those row types, treated as multivariate linear
# equations in the p,q values, have each coefficient being a value of
# the form \sum x^i, \sum y x^i or \sum y^2 x^i, for various i. (Times
# a factor of 2, but we can throw that away.) So we can go through the
# list of input coordinates summing all of those things, and then we
# have enough information to construct our matrix and solve it
# straight off for the rational function coefficients.

# Arguments:
#    f        The function to be approximated. Maps BigFloat -> BigFloat.
#    xvals    Array of BigFloats, giving the list of x-coordinates at which
#             to evaluate f.
#    n        Degree of the numerator polynomial of the desired rational
#             function.
#    d        Degree of the denominator polynomial of the desired rational
#             function.
#    w        Error-weighting function. Takes two BigFloat arguments x,y
#             and returns a scaling factor for the error at that location.
#             A larger value indicates that the error should be given
#             greater weight in the square sum we try to minimise.
#             If unspecified, defaults to giving everything the same weight.
#
# Return values: a pair of arrays of BigFloats (N,D) giving the
# coefficients of the returned rational function. N has size n+1; D
# has size d+1. Both start with the constant term, i.e. N[i] is the
# coefficient of x^(i-1) (because Julia arrays are 1-based). D[1] will
# be 1.
function ratfn_leastsquares(f::Function, xvals::Array{BigFloat}, n, d,
                            w = (x,y)->BigFloat(1))
    # Accumulate sums of x^i y^j, for j={0,1,2} and a range of x.
    # Again because Julia arrays are 1-based, we'll have sums[i,j]
    # being the sum of x^(i-1) y^(j-1).
    maxpow = max(n,d) * 2 + 1
    sums = zeros(BigFloat, maxpow, 3)
    for x = xvals
        y = f(x)
        weight = w(x,y)
        for i = 1:1:maxpow
            for j = 1:1:3
                sums[i,j] += x^(i-1) * y^(j-1) * weight
            end
        end
    end

    @debug("leastsquares", "sums=", repr(sums))

    # Build the matrix. We're solving n+d+1 equations in n+d+1
    # unknowns. (We actually have to return n+d+2 coefficients, but
    # one of them is hardwired to 1.)
    matrix = array2d(BigFloat, n+d+1, n+d+1)
    vector = array1d(BigFloat, n+d+1)
    for i = 0:1:n
        # Equation obtained by differentiating with respect to p_i,
        # i.e. the numerator coefficient of x^i.
        row = 1+i
        for j = 0:1:n
            matrix[row, 1+j] = sums[1+i+j, 1]
        end
        for j = 1:1:d
            matrix[row, 1+n+j] = -sums[1+i+j, 2]
        end
        vector[row] = sums[1+i, 2]
    end
    for i = 1:1:d
        # Equation obtained by differentiating with respect to q_i,
        # i.e. the denominator coefficient of x^i.
        row = 1+n+i
        for j = 0:1:n
            matrix[row, 1+j] = sums[1+i+j, 2]
        end
        for j = 1:1:d
            matrix[row, 1+n+j] = -sums[1+i+j, 3]
        end
        vector[row] = sums[1+i, 3]
    end

    @debug("leastsquares", "matrix=", repr(matrix))
    @debug("leastsquares", "vector=", repr(vector))

    # Solve the matrix equation.
    all_coeffs = matrix \ vector

    @debug("leastsquares", "all_coeffs=", repr(all_coeffs))

    # And marshal the results into two separate polynomial vectors to
    # return.
    ncoeffs = all_coeffs[1:n+1]
    dcoeffs = vcat([1], all_coeffs[n+2:n+d+1])
    return (ncoeffs, dcoeffs)
end

# ----------------------------------------------------------------------
# Golden-section search to find a maximum of a function.

# Arguments:
#    f        Function to be maximised/minimised. Maps BigFloat -> BigFloat.
#    a,b,c    BigFloats bracketing a maximum of the function.
#
# Expects:
#    a,b,c are in order (either a<=b<=c or c<=b<=a)
#    a != c             (but b can equal one or the other if it wants to)
#    f(a) <= f(b) >= f(c)
#
# Return value is an (x,y) pair of BigFloats giving the extremal input
# and output. (That is, y=f(x).)
function goldensection(f::Function, a::BigFloat, b::BigFloat, c::BigFloat)
    # Decide on a 'good enough' threshold.
    threshold = abs(c-a) * 2^(-epsbits/2)

    # We'll need the golden ratio phi, of course. Or rather, in this
    # case, we need 1/phi = 0.618...
    one_over_phi = 2 / (1 + sqrt(BigFloat(5)))

    # Flip round the interval endpoints so that the interval [a,b] is
    # at least as large as [b,c]. (Then we can always pick our new
    # point in [a,b] without having to handle lots of special cases.)
    if abs(b-a) < abs(c-a)
        a,  c  = c,  a
    end

    # Evaluate the function at the initial points.
    fa = f(a)
    fb = f(b)
    fc = f(c)

    @debug("goldensection", "starting")

    while abs(c-a) > threshold
        @debug("goldensection", "a: ", a, " -> ", fa)
        @debug("goldensection", "b: ", b, " -> ", fb)
        @debug("goldensection", "c: ", c, " -> ", fc)

        # Check invariants.
        @assert(a <= b <= c || c <= b <= a)
        @assert(fa <= fb >= fc)

        # Subdivide the larger of the intervals [a,b] and [b,c]. We've
        # arranged that this is always [a,b], for simplicity.
        d = a + (b-a) * one_over_phi

        # Now we have an interval looking like this (possibly
        # reversed):
        #
        #    a            d       b            c
        #
        # and we know f(b) is bigger than either f(a) or f(c). We have
        # two cases: either f(d) > f(b), or vice versa. In either
        # case, we can narrow to an interval of 1/phi the size, and
        # still satisfy all our invariants (three ordered points,
        # [a,b] at least the width of [b,c], f(a)<=f(b)>=f(c)).
        fd = f(d)
        @debug("goldensection", "d: ", d, " -> ", fd)
        if fd > fb
            a,  b,  c  = a,  d,  b
            fa, fb, fc = fa, fd, fb
            @debug("goldensection", "adb case")
        else
            a,  b,  c  = c,  b,  d
            fa, fb, fc = fc, fb, fd
            @debug("goldensection", "cbd case")
        end
    end

    @debug("goldensection", "done: ", b, " -> ", fb)
    return (b, fb)
end

# ----------------------------------------------------------------------
# Find the extrema of a function within a given interval.

# Arguments:
#    f         The function to be approximated. Maps BigFloat -> BigFloat.
#    grid      A set of points at which to evaluate f. Must be high enough
#              resolution to make extrema obvious.
#
# Returns an array of (x,y) pairs of BigFloats, with each x,y giving
# the extremum location and its value (i.e. y=f(x)).
function find_extrema(f::Function, grid::Array{BigFloat})
    len = length(grid)
    extrema = array1d(Tuple{BigFloat, BigFloat}, 0)
    for i = 1:1:len
        # We have to provide goldensection() with three points
        # bracketing the extremum. If the extremum is at one end of
        # the interval, then the only way we can do that is to set two
        # of the points equal (which goldensection() will cope with).
        prev = max(1, i-1)
        next = min(i+1, len)

        # Find our three pairs of (x,y) coordinates.
        xp, xi, xn = grid[prev], grid[i], grid[next]
        yp, yi, yn = f(xp), f(xi), f(xn)

        # See if they look like an extremum, and if so, ask
        # goldensection() to give a more exact location for it.
        if yp <= yi >= yn
            push!(extrema, goldensection(f, xp, xi, xn))
        elseif yp >= yi <= yn
            x, y = goldensection(x->-f(x), xp, xi, xn)
            push!(extrema, (x, -y))
        end
    end
    return extrema
end

# ----------------------------------------------------------------------
# Winnow a list of a function's extrema to give a subsequence of a
# specified length, with the extrema in the subsequence alternating
# signs, and with the smallest absolute value of an extremum in the
# subsequence as large as possible.
#
# We do this using a dynamic-programming approach. We work along the
# provided array of extrema, and at all times, we track the best set
# of extrema we have so far seen for each possible (length, sign of
# last extremum) pair. Each new extremum is evaluated to see whether
# it can be added to any previously seen best subsequence to make a
# new subsequence that beats the previous record holder in its slot.

# Arguments:
#    extrema   An array of (x,y) pairs of BigFloats giving the input extrema.
#    n         Number of extrema required as output.
#
# Returns a new array of (x,y) pairs which is a subsequence of the
# original sequence. (So, in particular, if the input was sorted by x
# then so will the output be.)
function winnow_extrema(extrema::Array{Tuple{BigFloat,BigFloat}}, n)
    # best[i,j] gives the best sequence so far of length i and with
    # sign j (where signs are coded as 1=positive, 2=negative), in the
    # form of a tuple (cost, actual array of x,y pairs).
    best = fill((BigFloat(0), array1d(Tuple{BigFloat,BigFloat}, 0)), n, 2)

    for (x,y) = extrema
        if y > 0
            sign = 1
        elseif y < 0
            sign = 2
        else
            # A zero-valued extremum cannot possibly contribute to any
            # optimal sequence, so we simply ignore it!
            continue
        end

        for i = 1:1:n
            # See if we can create a new entry for best[i,sign] by
            # appending our current (x,y) to some previous thing.
            if i == 1
                # Special case: we don't store a best zero-length
                # sequence :-)
                candidate = (abs(y), [(x,y)])
            else
                othersign = 3-sign # map 1->2 and 2->1
                oldscore, oldlist = best[i-1, othersign]
                newscore = min(abs(y), oldscore)
                newlist = vcat(oldlist, [(x,y)])
                candidate = (newscore, newlist)
            end
            # If our new candidate improves on the previous value of
            # best[i,sign], then replace it.
            if candidate[1] > best[i,sign][1]
                best[i,sign] = candidate
            end
        end
    end

    # Our ultimate return value has to be either best[n,1] or
    # best[n,2], but it could be either. See which one has the higher
    # score.
    if best[n,1][1] > best[n,2][1]
        ret = best[n,1][2]
    else
        ret = best[n,2][2]
    end
    # Make sure we did actually _find_ a good answer.
    @assert(length(ret) == n)
    return ret
end

# ----------------------------------------------------------------------
# Construct a rational-function approximation with equal and
# alternating weighted deviation at a specific set of x-coordinates.

# Arguments:
#    f         The function to be approximated. Maps BigFloat -> BigFloat.
#    coords    An array of BigFloats giving the x-coordinates. There should
#              be n+d+2 of them.
#    n, d      The degrees of the numerator and denominator of the desired
#              approximation.
#    prev_err  A plausible value for the alternating weighted deviation.
#              (Required to kickstart a binary search in the nonlinear case;
#              see comments below.)
#    w         Error-weighting function. Takes two BigFloat arguments x,y
#              and returns a scaling factor for the error at that location.
#              The returned approximation R should have the minimum possible
#              maximum value of abs((f(x)-R(x)) * w(x,f(x))). Optional
#              parameter, defaulting to the always-return-1 function.
#
# Return values: a pair of arrays of BigFloats (N,D) giving the
# coefficients of the returned rational function. N has size n+1; D
# has size d+1. Both start with the constant term, i.e. N[i] is the
# coefficient of x^(i-1) (because Julia arrays are 1-based). D[1] will
# be 1.
function ratfn_equal_deviation(f::Function, coords::Array{BigFloat},
                               n, d, prev_err::BigFloat,
                               w = (x,y)->BigFloat(1))
    @debug("equaldev", "n=", n, " d=", d, " coords=", repr(coords))
    @assert(length(coords) == n+d+2)

    if d == 0
        # Special case: we're after a polynomial. In this case, we
        # have the particularly easy job of just constructing and
        # solving a system of n+2 linear equations, to find the n+1
        # coefficients of the polynomial and also the amount of
        # deviation at the specified coordinates. Each equation is of
        # the form
        #
        #   p_0 x^0 + p_1 x^1 + ... + p_n x^n ± e/w(x) = f(x)
        #
        # in which the p_i and e are the variables, and the powers of
        # x and calls to w and f are the coefficients.

        matrix = array2d(BigFloat, n+2, n+2)
        vector = array1d(BigFloat, n+2)
        currsign = +1
        for i = 1:1:n+2
            x = coords[i]
            for j = 0:1:n
                matrix[i,1+j] = x^j
            end
            y = f(x)
            vector[i] = y
            matrix[i, n+2] = currsign / w(x,y)
            currsign = -currsign
        end

        @debug("equaldev", "matrix=", repr(matrix))
        @debug("equaldev", "vector=", repr(vector))

        outvector = matrix \ vector

        @debug("equaldev", "outvector=", repr(outvector))

        ncoeffs = outvector[1:n+1]
        dcoeffs = [BigFloat(1)]
        return ncoeffs, dcoeffs
    else
        # For a nontrivial rational function, the system of equations
        # we need to solve becomes nonlinear, because each equation
        # now takes the form
        #
        #   p_0 x^0 + p_1 x^1 + ... + p_n x^n
        #   --------------------------------- ± e/w(x) = f(x)
        #     x^0 + q_1 x^1 + ... + q_d x^d
        #
        # and multiplying up by the denominator gives you a lot of
        # terms containing e × q_i. So we can't do this the really
        # easy way using a matrix equation as above.
        #
        # Fortunately, this is a fairly easy kind of nonlinear system.
        # The equations all become linear if you switch to treating e
        # as a constant, so a reasonably sensible approach is to pick
        # a candidate value of e, solve all but one of the equations
        # for the remaining unknowns, and then see what the error
        # turns out to be in the final equation. The Chebyshev
        # alternation theorem guarantees that that error in the last
        # equation will be anti-monotonic in the input e, so we can
        # just binary-search until we get the two as close to equal as
        # we need them.

        function try_e(e)
            # Try a given value of e, derive the coefficients of the
            # resulting rational function by setting up equations
            # based on the first n+d+1 of the n+d+2 coordinates, and
            # see what the error turns out to be at the final
            # coordinate.
            matrix = array2d(BigFloat, n+d+1, n+d+1)
            vector = array1d(BigFloat, n+d+1)
            currsign = +1
            for i = 1:1:n+d+1
                x = coords[i]
                y = f(x)
                y_adj = y - currsign * e / w(x,y)
                for j = 0:1:n
                    matrix[i,1+j] = x^j
                end
                for j = 1:1:d
                    matrix[i,1+n+j] = -x^j * y_adj
                end
                vector[i] = y_adj
                currsign = -currsign
            end

            @debug("equaldev", "trying e=", e)
            @debug("equaldev", "matrix=", repr(matrix))
            @debug("equaldev", "vector=", repr(vector))

            outvector = matrix \ vector

            @debug("equaldev", "outvector=", repr(outvector))

            ncoeffs = outvector[1:n+1]
            dcoeffs = vcat([BigFloat(1)], outvector[n+2:n+d+1])

            x = coords[n+d+2]
            y = f(x)
            last_e = (ratfn_eval(ncoeffs, dcoeffs, x) - y) * w(x,y) * -currsign

            @debug("equaldev", "last e=", last_e)

            return ncoeffs, dcoeffs, last_e
        end

        threshold = 2^(-epsbits/2) # convergence threshold

        # Start by trying our previous iteration's error value. This
        # value (e0) will be one end of our binary-search interval,
        # and whatever it caused the last point's error to be, that
        # (e1) will be the other end.
        e0 = prev_err
        @debug("equaldev", "e0 = ", e0)
        nc, dc, e1 = try_e(e0)
        @debug("equaldev", "e1 = ", e1)
        if abs(e1-e0) <= threshold
            # If we're _really_ lucky, we hit the error right on the
            # nose just by doing that!
            return nc, dc
        end
        s = sign(e1-e0)
        @debug("equaldev", "s = ", s)

        # Verify by assertion that trying our other interval endpoint
        # e1 gives a value that's wrong in the other direction.
        # (Otherwise our binary search won't get a sensible answer at
        # all.)
        nc, dc, e2 = try_e(e1)
        @debug("equaldev", "e2 = ", e2)
        @assert(sign(e2-e1) == -s)

        # Now binary-search until our two endpoints narrow enough.
        local emid
        while abs(e1-e0) > threshold
            emid = (e1+e0)/2
            nc, dc, enew = try_e(emid)
            if sign(enew-emid) == s
                e0 = emid
            else
                e1 = emid
            end
        end

        @debug("equaldev", "final e=", emid)
        return nc, dc
    end
end

# ----------------------------------------------------------------------
# Top-level function to find a minimax rational-function approximation.

# Arguments:
#    f         The function to be approximated. Maps BigFloat -> BigFloat.
#    interval  A pair of BigFloats giving the endpoints of the interval
#              (in either order) on which to approximate f.
#    n, d      The degrees of the numerator and denominator of the desired
#              approximation.
#    w         Error-weighting function. Takes two BigFloat arguments x,y
#              and returns a scaling factor for the error at that location.
#              The returned approximation R should have the minimum possible
#              maximum value of abs((f(x)-R(x)) * w(x,f(x))). Optional
#              parameter, defaulting to the always-return-1 function.
#
# Return values: a tuple (N,D,E,X), where

#    N,D       A pair of arrays of BigFloats giving the coefficients
#              of the returned rational function. N has size n+1; D
#              has size d+1. Both start with the constant term, i.e.
#              N[i] is the coefficient of x^(i-1) (because Julia
#              arrays are 1-based). D[1] will be 1.
#    E         The maximum weighted error (BigFloat).
#    X         An array of pairs of BigFloats giving the locations of n+2
#              points and the weighted error at each of those points. The
#              weighted error values will have alternating signs, which
#              means that the Chebyshev alternation theorem guarantees
#              that any other function of the same degree must exceed
#              the error of this one at at least one of those points.
function ratfn_minimax(f::Function, interval::Tuple{BigFloat,BigFloat}, n, d,
                       w = (x,y)->BigFloat(1))
    # We start off by finding a least-squares approximation. This
    # doesn't need to be perfect, but if we can get it reasonably good
    # then it'll save iterations in the refining stage.
    #
    # Least-squares approximations tend to look nicer in a minimax
    # sense if you evaluate the function at a big pile of Chebyshev
    # nodes rather than uniformly spaced points. These values will
    # also make a good grid to use for the initial search for error
    # extrema, so we'll keep them around for that reason too.

    # Construct the grid.
    lo, hi = minimum(interval), maximum(interval)
    local grid
    let
        mid = (hi+lo)/2
        halfwid = (hi-lo)/2
        nnodes = 16 * (n+d+1)
        pi = 2*asin(BigFloat(1))
        grid = [ mid - halfwid * cos(pi*i/nnodes) for i=0:1:nnodes ]
    end

    # Find the initial least-squares approximation.
    (nc, dc) = ratfn_leastsquares(f, grid, n, d, w)
    @debug("minimax", "initial leastsquares approx = ",
           ratfn_to_string(nc, dc))

    # Threshold of convergence. We stop when the relative difference
    # between the min and max (winnowed) error extrema is less than
    # this.
    #
    # This is set to the cube root of machine epsilon on a more or
    # less empirical basis, because the rational-function case will
    # not converge reliably if you set it to only the square root.
    # (Repeatable by using the --test mode.) On the assumption that
    # input and output error in each iteration can be expected to be
    # related by a simple power law (because it'll just be down to how
    # many leading terms of a Taylor series are zero), the cube root
    # was the next thing to try.
    threshold = 2^(-epsbits/3)

    # Main loop.
    while true
        # Find all the error extrema we can.
        function compute_error(x)
            real_y = f(x)
            approx_y = ratfn_eval(nc, dc, x)
            return (approx_y - real_y) * w(x, real_y)
        end
        extrema = find_extrema(compute_error, grid)
        @debug("minimax", "all extrema = ", format_xylist(extrema))

        # Winnow the extrema down to the right number, and ensure they
        # have alternating sign.
        extrema = winnow_extrema(extrema, n+d+2)
        @debug("minimax", "winnowed extrema = ", format_xylist(extrema))

        # See if we've finished.
        min_err = minimum([abs(y) for (x,y) = extrema])
        max_err = maximum([abs(y) for (x,y) = extrema])
        variation = (max_err - min_err) / max_err
        @debug("minimax", "extremum variation = ", variation)
        if variation < threshold
            @debug("minimax", "done!")
            return nc, dc, max_err, extrema
        end

        # If not, refine our function by equalising the error at the
        # extrema points, and go round again.
        (nc, dc) = ratfn_equal_deviation(f, map(x->x[1], extrema),
                                         n, d, max_err, w)
        @debug("minimax", "refined approx = ", ratfn_to_string(nc, dc))
    end
end

# ----------------------------------------------------------------------
# Check if a polynomial is well-conditioned for accurate evaluation in
# a given interval by Horner's rule.
#
# This is true if at every step where Horner's rule computes
# (coefficient + x*value_so_far), the constant coefficient you're
# adding on is of larger magnitude than the x*value_so_far operand.
# And this has to be true for every x in the interval.
#
# Arguments:
#    coeffs    The coefficients of the polynomial under test. Starts with
#              the constant term, i.e. coeffs[i] is the coefficient of
#              x^(i-1) (because Julia arrays are 1-based).
#    lo, hi    The bounds of the interval.
#
# Return value: the largest ratio (x*value_so_far / coefficient), at
# any step of evaluation, for any x in the interval. If this is less
# than 1, the polynomial is at least somewhat well-conditioned;
# ideally you want it to be more like 1/8 or 1/16 or so, so that the
# relative rounding error accumulated at each step are reduced by
# several factors of 2 when the next coefficient is added on.

function wellcond(coeffs, lo, hi)
    x = max(abs(lo), abs(hi))
    worst = 0
    so_far = 0
    for i = length(coeffs):-1:1
        coeff = abs(coeffs[i])
        so_far *= x
        if coeff != 0
            thisval = so_far / coeff
            worst = max(worst, thisval)
            so_far += coeff
        end
    end
    return worst
end

# ----------------------------------------------------------------------
# Small set of unit tests.

function test()
    passes = 0
    fails = 0

    function approx_eq(x, y, limit=1e-6)
        return abs(x - y) < limit
    end

    function test(condition)
        if condition
            passes += 1
        else
            println("fail")
            fails += 1
        end
    end

    # Test Gaussian elimination.
    println("Gaussian test 1:")
    m = BigFloat[1 1 2; 3 5 8; 13 34 21]
    v = BigFloat[1, -1, 2]
    ret = m \ v
    println("  ",repr(ret))
    test(approx_eq(ret[1], 109/26))
    test(approx_eq(ret[2], -105/130))
    test(approx_eq(ret[3], -31/26))

    # Test leastsquares rational functions.
    println("Leastsquares test 1:")
    n = 10000
    a = array1d(BigFloat, n+1)
    for i = 0:1:n
        a[1+i] = i/BigFloat(n)
    end
    (nc, dc) = ratfn_leastsquares(x->exp(x), a, 2, 2)
    println("  ",ratfn_to_string(nc, dc))
    for x = a
        test(approx_eq(exp(x), ratfn_eval(nc, dc, x), 1e-4))
    end

    # Test golden section search.
    println("Golden section test 1:")
    x, y = goldensection(x->sin(x),
                              BigFloat(0), BigFloat(1)/10, BigFloat(4))
    println("  ", x, " -> ", y)
    test(approx_eq(x, asin(BigFloat(1))))
    test(approx_eq(y, 1))

    # Test extrema-winnowing algorithm.
    println("Winnow test 1:")
    extrema = [(x, sin(20*x)*sin(197*x))
               for x in BigFloat(0):BigFloat(1)/1000:BigFloat(1)]
    winnowed = winnow_extrema(extrema, 12)
    println("  ret = ", format_xylist(winnowed))
    prevx, prevy = -1, 0
    for (x,y) = winnowed
        test(x > prevx)
        test(y != 0)
        test(prevy * y <= 0) # tolerates initial prevx having no sign
        test(abs(y) > 0.9)
        prevx, prevy = x, y
    end

    # Test actual minimax approximation.
    println("Minimax test 1 (polynomial):")
    (nc, dc, e, x) = ratfn_minimax(x->exp(x), (BigFloat(0), BigFloat(1)), 4, 0)
    println("  ",e)
    println("  ",ratfn_to_string(nc, dc))
    test(0 < e < 1e-3)
    for x = 0:BigFloat(1)/1000:1
        test(abs(ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001)
    end

    println("Minimax test 2 (rational):")
    (nc, dc, e, x) = ratfn_minimax(x->exp(x), (BigFloat(0), BigFloat(1)), 2, 2)
    println("  ",e)
    println("  ",ratfn_to_string(nc, dc))
    test(0 < e < 1e-3)
    for x = 0:BigFloat(1)/1000:1
        test(abs(ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001)
    end

    println("Minimax test 3 (polynomial, weighted):")
    (nc, dc, e, x) = ratfn_minimax(x->exp(x), (BigFloat(0), BigFloat(1)), 4, 0,
                                   (x,y)->1/y)
    println("  ",e)
    println("  ",ratfn_to_string(nc, dc))
    test(0 < e < 1e-3)
    for x = 0:BigFloat(1)/1000:1
        test(abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001)
    end

    println("Minimax test 4 (rational, weighted):")
    (nc, dc, e, x) = ratfn_minimax(x->exp(x), (BigFloat(0), BigFloat(1)), 2, 2,
                                   (x,y)->1/y)
    println("  ",e)
    println("  ",ratfn_to_string(nc, dc))
    test(0 < e < 1e-3)
    for x = 0:BigFloat(1)/1000:1
        test(abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001)
    end

    println("Minimax test 5 (rational, weighted, odd degree):")
    (nc, dc, e, x) = ratfn_minimax(x->exp(x), (BigFloat(0), BigFloat(1)), 2, 1,
                                   (x,y)->1/y)
    println("  ",e)
    println("  ",ratfn_to_string(nc, dc))
    test(0 < e < 1e-3)
    for x = 0:BigFloat(1)/1000:1
        test(abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001)
    end

    total = passes + fails
    println(passes, " passes ", fails, " fails ", total, " total")
end

# ----------------------------------------------------------------------
# Online help.
function help()
    print("""
Usage:

    remez.jl [options] <lo> <hi> <n> <d> <expr> [<weight>]

Arguments:

    <lo>, <hi>

        Bounds of the interval on which to approximate the target
        function. These are parsed and evaluated as Julia expressions,
        so you can write things like '1/BigFloat(6)' to get an
        accurate representation of 1/6, or '4*atan(BigFloat(1))' to
        get pi. (Unfortunately, the obvious 'BigFloat(pi)' doesn't
        work in Julia.)

    <n>, <d>

        The desired degree of polynomial(s) you want for your
        approximation. These should be non-negative integers. If you
        want a rational function as output, set <n> to the degree of
        the numerator, and <d> the denominator. If you just want an
        ordinary polynomial, set <d> to 0, and <n> to the degree of
        the polynomial you want.

    <expr>

        A Julia expression giving the function to be approximated on
        the interval. The input value is predefined as 'x' when this
        expression is evaluated, so you should write something along
        the lines of 'sin(x)' or 'sqrt(1+tan(x)^2)' etc.

    <weight>

        If provided, a Julia expression giving the weighting factor
        for the approximation error. The output polynomial will
        minimise the largest absolute value of (P-f) * w at any point
        in the interval, where P is the value of the polynomial, f is
        the value of the target function given by <expr>, and w is the
        weight given by this function.

        When this expression is evaluated, the input value to P and f
        is predefined as 'x', and also the true output value f(x) is
        predefined as 'y'. So you can minimise the relative error by
        simply writing '1/y'.

        If the <weight> argument is not provided, the default
        weighting function always returns 1, so that the polynomial
        will minimise the maximum absolute error |P-f|.

Computation options:

    --pre=<predef_expr>

        Evaluate the Julia expression <predef_expr> before starting
        the computation. This permits you to pre-define variables or
        functions which the Julia expressions in your main arguments
        can refer to. All of <lo>, <hi>, <expr> and <weight> can make
        use of things defined by <predef_expr>.

        One internal remez.jl function that you might sometimes find
        useful in this expression is 'goldensection', which finds the
        location and value of a maximum of a function. For example,
        one implementation strategy for the gamma function involves
        translating it to put its unique local minimum at the origin,
        in which case you can write something like this

            --pre='(m,my) = goldensection(x -> -gamma(x),
                  BigFloat(1), BigFloat(1.5), BigFloat(2))'

        to predefine 'm' as the location of gamma's minimum, and 'my'
        as the (negated) value that gamma actually takes at that
        point, i.e. -gamma(m).

        (Since 'goldensection' always finds a maximum, we had to
        negate gamma in the input function to make it find a minimum
        instead. Consult the comments in the source for more details
        on the use of this function.)

        If you use this option more than once, all the expressions you
        provide will be run in sequence.

    --bits=<bits>

        Specify the accuracy to which you want the output polynomial,
        in bits. Default 256, which should be more than enough.

    --bigfloatbits=<bits>

        Turn up the precision used by Julia for its BigFloat
        evaluation. Default is Julia's default (also 256). You might
        want to try setting this higher than the --bits value if the
        algorithm is failing to converge for some reason.

Output options:

    --full

        Instead of just printing the approximation function itself,
        also print auxiliary information:
         - the locations of the error extrema, and the actual
           (weighted) error at each of those locations
         - the overall maximum error of the function
         - a 'well-conditioning quotient', giving the worst-case ratio
           between any polynomial coefficient and the largest possible
           value of the higher-order terms it will be added to.

        The well-conditioning quotient should be less than 1, ideally
        by several factors of two, for accurate evaluation in the
        target precision. If you request a rational function, a
        separate well-conditioning quotient will be printed for the
        numerator and denominator.

        Use this option when deciding how wide an interval to
        approximate your function on, and what degree of polynomial
        you need.

    --variable=<identifier>

        When writing the output polynomial or rational function in its
        usual form as an arithmetic expression, use <identifier> as
        the name of the input variable. Default is 'x'.

    --suffix=<suffix>

        When writing the output polynomial or rational function in its
        usual form as an arithmetic expression, write <suffix> after
        every floating-point literal. For example, '--suffix=F' will
        generate a C expression in which the coefficients are literals
        of type 'float' rather than 'double'.

    --array

        Instead of writing the output polynomial as an arithmetic
        expression in Horner's rule form, write out just its
        coefficients, one per line, each with a trailing comma.
        Suitable for pasting into a C array declaration.

        This option is not currently supported if the output is a
        rational function, because you'd need two separate arrays for
        the numerator and denominator coefficients and there's no
        obviously right way to provide both of those together.

Debug and test options:

    --debug=<facility>

        Enable debugging output from various parts of the Remez
        calculation. <facility> should be the name of one of the
        classes of diagnostic output implemented in the program.
        Useful values include 'gausselim', 'leastsquares',
        'goldensection', 'equaldev', 'minimax'. This is probably
        mostly useful to people debugging problems with the script, so
        consult the source code for more information about what the
        diagnostic output for each of those facilities will be.

        If you want diagnostics from more than one facility, specify
        this option multiple times with different arguments.

    --test

        Run remez.jl's internal test suite. No arguments needed.

Miscellaneous options:

    --help

        Display this text and exit. No arguments needed.

""")
end

# ----------------------------------------------------------------------
# Main program.

function main()
    nargs = length(argwords)
    if nargs != 5 && nargs != 6
        error("usage: remez.jl <lo> <hi> <n> <d> <expr> [<weight>]\n" *
              "       run 'remez.jl --help' for more help")
    end

    for preliminary_command in preliminary_commands
        eval(Meta.parse(preliminary_command))
    end

    lo = BigFloat(eval(Meta.parse(argwords[1])))
    hi = BigFloat(eval(Meta.parse(argwords[2])))
    n = parse(Int,argwords[3])
    d = parse(Int,argwords[4])
    f = eval(Meta.parse("x -> " * argwords[5]))

    # Wrap the user-provided function with a function of our own. This
    # arranges to detect silly FP values (inf,nan) early and diagnose
    # them sensibly, and also lets us log all evaluations of the
    # function in case you suspect it's doing the wrong thing at some
    # special-case point.
    function func(x)
        y = run(f,x)
        @debug("f", x, " -> ", y)
        if !isfinite(y)
            error("f(" * string(x) * ") returned non-finite value " * string(y))
        end
        return y
    end

    if nargs == 6
        # Wrap the user-provided weight function similarly.
        w = eval(Meta.parse("(x,y) -> " * argwords[6]))
        function wrapped_weight(x,y)
            ww = run(w,x,y)
            if !isfinite(ww)
                error("w(" * string(x) * "," * string(y) *
                      ") returned non-finite value " * string(ww))
            end
            return ww
        end
        weight = wrapped_weight
    else
        weight = (x,y)->BigFloat(1)
    end

    (nc, dc, e, extrema) = ratfn_minimax(func, (lo, hi), n, d, weight)
    if array_format
        if d == 0
            functext = join([string(x)*",\n" for x=nc],"")
        else
            # It's unclear how you should best format an array of
            # coefficients for a rational function, so I'll leave
            # implementing this option until I have a use case.
            error("--array unsupported for rational functions")
        end
    else
        functext = ratfn_to_string(nc, dc) * "\n"
    end
    if full_output
        # Print everything you might want to know about the function
        println("extrema = ", format_xylist(extrema))
        println("maxerror = ", string(e))
        if length(dc) > 1
            println("wellconditioning_numerator = ",
                    string(wellcond(nc, lo, hi)))
            println("wellconditioning_denominator = ",
                    string(wellcond(dc, lo, hi)))
        else
            println("wellconditioning = ", string(wellcond(nc, lo, hi)))
        end
        print("function = ", functext)
    else
        # Just print the text people will want to paste into their code
        print(functext)
    end
end

# ----------------------------------------------------------------------
# Top-level code: parse the argument list and decide what to do.

what_to_do = main

doing_opts = true
argwords = array1d(String, 0)
for arg = ARGS
    global doing_opts, what_to_do, argwords
    global full_output, array_format, xvarname, floatsuffix, epsbits
    if doing_opts && startswith(arg, "-")
        if arg == "--"
            doing_opts = false
        elseif arg == "--help"
            what_to_do = help
        elseif arg == "--test"
            what_to_do = test
        elseif arg == "--full"
            full_output = true
        elseif arg == "--array"
            array_format = true
        elseif startswith(arg, "--debug=")
            enable_debug(arg[length("--debug=")+1:end])
        elseif startswith(arg, "--variable=")
            xvarname = arg[length("--variable=")+1:end]
        elseif startswith(arg, "--suffix=")
            floatsuffix = arg[length("--suffix=")+1:end]
        elseif startswith(arg, "--bits=")
            epsbits = parse(Int,arg[length("--bits=")+1:end])
        elseif startswith(arg, "--bigfloatbits=")
            set_bigfloat_precision(
                parse(Int,arg[length("--bigfloatbits=")+1:end]))
        elseif startswith(arg, "--pre=")
            push!(preliminary_commands, arg[length("--pre=")+1:end])
        else
            error("unrecognised option: ", arg)
        end
    else
        push!(argwords, arg)
    end
end

what_to_do()
