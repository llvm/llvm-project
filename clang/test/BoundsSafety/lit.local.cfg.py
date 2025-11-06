import lit.formats
import os

# clang/test uses external shell by default, but all tests in clang/test/BoundsSafety
# support lit's internal shell, so use that instead
config.test_format = lit.formats.ShTest(False)

cc1_substituted = [ False ]
clang_substituted = [ False ]
extra_flags = '-fbounds-safety-bringup-missing-checks=all'

def get_subst(subst_name):
    subst_re = r'%\b{}\b'.format(subst_name)
    for subst_name, value in config.substitutions:
        if subst_re not in subst_name:
            continue
        return value
    raise Exception(f'Could not find %{subst_name} substitution')

def patch_subst_impl(sub_tuple, subst_to_patch):
    assert isinstance(sub_tuple, tuple)
    assert len(sub_tuple) == 2
    subst_name = sub_tuple[0]

    # Match `%subst_name` (e.g. `%clang_cc1`). The `ToolSubst` class inside
    # `lit` adds a bunch of regexes around this substitution so we try to match
    # them here to avoid matching a substitution that has `%clang_cc1` as a
    # substring. This is fragile
    subst_re = r'%\b{}\b'.format(subst_to_patch)
    if subst_re not in subst_name:
        return (sub_tuple[0], sub_tuple[1], False)
    patched_sub = sub_tuple[1] + f' {extra_flags}'
    return (sub_tuple[0], patched_sub, True)

def patch_cc1_subst(sub_tuple):
    subst_name, subst, patched = patch_subst_impl(sub_tuple, 'clang_cc1')
    if patched:
        cc1_substituted[0] = True
    return (subst_name, subst)

def patch_clang_subst(sub_tuple):
    subst_name, subst, patched = patch_subst_impl(sub_tuple, 'clang')
    if patched:
        clang_substituted[0] = True
    return (subst_name, subst)


force_new_bounds_checks_on = False

# Provide an un-patched substitution
default_clang_subst = get_subst('clang')
config.substitutions.append(
    (r'%\bclang_no_bounds_check_mode_specified\b', default_clang_subst)
)

if False:
    force_new_bounds_checks_on = True

if force_new_bounds_checks_on:
    config.substitutions = list(map(patch_cc1_subst, config.substitutions))
    config.substitutions = list(map(patch_clang_subst, config.substitutions))

    dir_with_patched_sub = os.path.dirname(__file__)
    for subst, patched in [('%clang_cc1', cc1_substituted[0]), ('%clang', clang_substituted[0])]:
        if not patched:
            lit_config.fatal(f'Failed to add `{extra_flags}` to {subst} invocations under {dir_with_patched_sub}')
        else:
            lit_config.note(f'Implicitly passing `{extra_flags}` to {subst} invocations under {dir_with_patched_sub}')

