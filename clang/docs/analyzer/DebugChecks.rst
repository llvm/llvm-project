============
Debug Checks
============

.. contents::
   :local:

The analyzer contains a number of checkers which can aid in debugging. Enable
them by using the "-analyzer-checker=" flag, followed by the name of the
checker.


General Analysis Dumpers
========================

These checkers are used to dump the results of various infrastructural analyses
to stderr. Some checkers also have "view" variants, which will display a graph
using a 'dot' format viewer (such as Graphviz on OS X) instead.

- debug.DumpCallGraph, debug.ViewCallGraph: Show the call graph generated for
  the current translation unit. This is used to determine the order in which to
  analyze functions when inlining is enabled.

- debug.DumpCFG, debug.ViewCFG: Show the CFG generated for each top-level
  function being analyzed.

- debug.DumpDominators: Shows the dominance tree for the CFG of each top-level
  function.

- debug.DumpLiveVars: Show the results of live variable analysis for each
  top-level function being analyzed.

- debug.ViewExplodedGraph: Show the Exploded Graphs generated for the
  analysis of different functions in the input translation unit. When there
  are several functions analyzed, display one graph per function. Beware 
  that these graphs may grow very large, even for small functions.

Path Tracking
=============

These checkers print information about the path taken by the analyzer engine.

- debug.DumpCalls: Prints out every function or method call encountered during a
  path traversal. This is indented to show the call stack, but does NOT do any
  special handling of branches, meaning different paths could end up
  interleaved.

- debug.DumpTraversal: Prints the name of each branch statement encountered
  during a path traversal ("IfStmt", "WhileStmt", etc). Currently used to check
  whether the analysis engine is doing BFS or DFS.


State Checking
==============

These checkers will print out information about the analyzer state in the form
of analysis warnings. They are intended for use with the -verify functionality
in regression tests.

- debug.TaintTest: Prints out the word "tainted" for every expression that
  carries taint. At the time of this writing, taint was only introduced by the
  checks under experimental.security.taint.TaintPropagation; this checker may
  eventually move to the security.taint package.

- debug.ExprInspection: Responds to certain function calls, which are modeled
  after builtins. These function calls should affect the program state other
  than the evaluation of their arguments; to use them, you will need to declare
  them within your test file. The available functions are described below.

(FIXME: debug.ExprInspection should probably be renamed, since it no longer only
inspects expressions.)


ExprInspection checks
---------------------

- void clang_analyzer_eval(bool);

  Prints TRUE if the argument is known to have a non-zero value, FALSE if the
  argument is known to have a zero or null value, and UNKNOWN if the argument
  isn't sufficiently constrained on this path.  You can use this to test other
  values by using expressions like "x == 5".  Note that this functionality is
  currently DISABLED in inlined functions, since different calls to the same
  inlined function could provide different information, making it difficult to
  write proper -verify directives.

  In C, the argument can be typed as 'int' or as '_Bool'.

  Example usage::

    clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
    if (!x) return;
    clang_analyzer_eval(x); // expected-warning{{TRUE}}


- void clang_analyzer_checkInlined(bool);

  If a call occurs within an inlined function, prints TRUE or FALSE according to
  the value of its argument. If a call occurs outside an inlined function,
  nothing is printed.

  The intended use of this checker is to assert that a function is inlined at
  least once (by passing 'true' and expecting a warning), or to assert that a
  function is never inlined (by passing 'false' and expecting no warning). The
  argument is technically unnecessary but is intended to clarify intent.

  You might wonder why we can't print TRUE if a function is ever inlined and
  FALSE if it is not. The problem is that any inlined function could conceivably
  also be analyzed as a top-level function (in which case both TRUE and FALSE
  would be printed), depending on the value of the -analyzer-inlining option.

  In C, the argument can be typed as 'int' or as '_Bool'.

  Example usage::

    int inlined() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return 42;
    }
    
    void topLevel() {
      clang_analyzer_checkInlined(false); // no-warning (not inlined)
      int value = inlined();
      // This assertion will not be valid if the previous call was not inlined.
      clang_analyzer_eval(value == 42); // expected-warning{{TRUE}}
    }

- void clang_analyzer_warnIfReached();

  Generate a warning if this line of code gets reached by the analyzer.

  Example usage::

    if (true) {
      clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    }
    else {
      clang_analyzer_warnIfReached();  // no-warning
    }


Statistics
==========

The debug.Stats checker collects various information about the analysis of each
function, such as how many blocks were reached and if the analyzer timed out.

There is also an additional -analyzer-stats flag, which enables various
statistics within the analyzer engine. Note the Stats checker (which produces at
least one bug report per function) may actually change the values reported by
-analyzer-stats.
