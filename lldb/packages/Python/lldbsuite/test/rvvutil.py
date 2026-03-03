"""
This LLDB module contains RISC-V RVV test utilities.
"""

import lldb


def skip_if_rvv_unsupported(test):
    if not test.isRISCVRVV():
        test.skipTest("RVV registers must be supported.")


def skip_if_rvv_supported(test):
    if test.isRISCVRVV():
        test.skipTest("RVV registers must be unsupported.")


def get_register_value(test, reg_name):
    """Get a reg_name register value."""
    frame = test.thread().GetFrameAtIndex(0)
    reg = frame.FindRegister(reg_name)
    if not reg.IsValid():
        return None
    error = lldb.SBError()
    return reg.GetValueAsUnsigned(error)


def get_vlenb(test):
    """Get vlenb register value."""
    return get_register_value(test, "vlenb")


def get_lmul(test):
    vtype = get_register_value(test, "vtype")
    lmul_code = vtype & 0x7
    if lmul_code == 4:
        return None

    is_frac = lmul_code > 3
    lmul = 1 / (1 << (8 - lmul_code)) if is_frac else 1 << lmul_code
    return lmul


def get_sew(test):
    vtype = get_register_value(test, "vtype")
    sew_code = (vtype >> 3) & 0x7
    if sew_code >= 4:
        return None

    sew = 8 << sew_code
    return sew


def calculate_vlmax(test):
    vlenb = get_vlenb(test)
    vlmax = vlenb * 8 / get_sew(test) * get_lmul(test)
    return round(vlmax)


def set_vector_register_bytes(test, reg_name, byte_list):
    """Set vector register to specific byte values."""
    byte_str = "{" + " ".join([f"0x{b:02x}" for b in byte_list]) + "}"
    test.runCmd(f"register write {reg_name} '{byte_str}'")


def check_vector_register_bytes(test, reg_name, expected_bytes):
    """Check that vector register contains expected bytes."""
    byte_str = "{" + " ".join([f"0x{b:02x}" for b in expected_bytes]) + "}"
    test.expect(f"register read {reg_name}", substrs=[f"{reg_name} = {byte_str}"])
