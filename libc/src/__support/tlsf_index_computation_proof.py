import sys
import math

try:
    # Try importing cvc5's pythonic API wrapper as z3 (drop-in replacement)
    import cvc5.pythonic as z3
    # Verify it is the correct pythonic module and not the base cvc5 module
    _ = z3.BitVec
    SOLVER_NAME = "cvc5 (pythonic API)"
except (ImportError, AttributeError):
    try:
        import z3
        SOLVER_NAME = "Z3"
    except ImportError:
        print("Error: Neither cvc5 nor z3-solver is installed.")
        print("Please install one of them:")
        print("  pip install cvc5")
        print("  pip install z3-solver")
        sys.exit(1)

def log2_z3(y):
    # Returns a symbolic expression representing floor(log2(y)) for y >= 1.
    # We build a nested structure of conditional updates to find the highest set bit.
    expr = z3.BitVecVal(0, 64)
    for i in range(1, 64):
        expr = z3.If(z3.LShR(y, i) == 1, z3.BitVecVal(i, 64), expr)
    return expr

def prove_equivalence(unit_size, step_size_bits, num_step_bits, total_bits):
    print(f"Verifying TLSF configuration using {SOLVER_NAME}:")
    print(f"  UNIT_SIZE        = {unit_size}")
    print(f"  STEP_SIZE_BITS   = {step_size_bits}")
    print(f"  NUM_STEP_BITS    = {num_step_bits}")
    print(f"  TOTAL_BITS       = {total_bits}")

    step_size = 1 << step_size_bits
    num_steps = 1 << num_step_bits
    exp_base = step_size * num_steps
    exp_base_log2 = step_size_bits + num_step_bits

    unit_size_log2 = int(math.log2(unit_size))

    # The raw size is the primary 64-bit input parameter
    size = z3.BitVec('size', 64)
    
    # x represents the shifted_size (size >> unit_size_log2)
    x = z3.LShR(size, unit_size_log2)

    # -------------------------------------------------------------------------
    # 1. Raw Branchy Algorithm
    # -------------------------------------------------------------------------
    # exp_index = log2(x / EXP_BASE)
    exp_index = log2_z3(z3.UDiv(x, exp_base))
    
    base_shifted = z3.BitVecVal(exp_base, 64) << exp_index
    step_shifted = base_shifted >> num_step_bits
    
    # linear_index = (x - base_shifted) / step_shifted
    linear_index = z3.UDiv(x - base_shifted, step_shifted)
    
    index_raw = z3.BitVecVal(exp_base, 64) + z3.BitVecVal(num_steps, 64) * exp_index + linear_index
    clipped_raw = z3.If(index_raw < total_bits, index_raw, z3.BitVecVal(total_bits - 1, 64))

    # -------------------------------------------------------------------------
    # 2. Optimized Shift-and-Add Algorithm (Direct Path, No Pre-shifting)
    # -------------------------------------------------------------------------
    size_ilog2 = log2_z3(size)

    # Offset term: (size_ilog2 - unit_size_log2 - exp_base_log2 - 1)
    offset_term = size_ilog2 - unit_size_log2 - exp_base_log2 - 1
    exp_offset = offset_term << num_step_bits
    
    # step_index = size >> (size_ilog2 - num_step_bits)
    step_index = z3.LShR(size, size_ilog2 - num_step_bits)
    
    index_opt = z3.BitVecVal(exp_base, 64) + exp_offset + step_index
    clipped_opt = z3.If(index_opt < total_bits, index_opt, z3.BitVecVal(total_bits - 1, 64))

    # -------------------------------------------------------------------------
    # Equivalence Proof Solver Setup
    # -------------------------------------------------------------------------
    s = z3.Solver()
    
    # Assert precondition: size must be strictly within the large sizes range
    s.add(z3.UGT(size, z3.BitVecVal(exp_base << unit_size_log2, 64)))
    
    # Query if there exists ANY input size where the two calculations diverge:
    s.add(clipped_raw != clipped_opt)
    
    print("Solving for logical divergence/equivalence...")
    result = s.check()
    if result == z3.unsat:
        print("\n[SUCCESS] Both algorithms are FORMALLY EQUIVALENT for all inputs!")
    elif result == z3.sat:
        print("\n[FAILURE] Found a logical divergence counter-example!")
        m = s.model()
        val_size = m[size].as_long()
        print(f"  For size = {val_size} (shifted_size = {val_size >> unit_size_log2}):")
        raw_val = s.model().evaluate(clipped_raw).as_long()
        opt_val = s.model().evaluate(clipped_opt).as_long()
        print(f"    Raw Index: {raw_val}")
        print(f"    Opt Index: {opt_val}")
    else:
        print("\n[UNKNOWN] Solver returned unknown. Could not verify.")

if __name__ == "__main__":
    # TLSF Project parameters (UNIT_SIZE=16, STEP_SIZE_BITS=2, NUM_STEP_BITS=4, TOTAL_BITS=256)
    prove_equivalence(unit_size=16, step_size_bits=2, num_step_bits=4, total_bits=256)
