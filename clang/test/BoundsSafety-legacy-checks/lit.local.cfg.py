import os

cc1_substituted = [ False ]
extra_flags = '-fno-bounds-safety-bringup-missing-checks=all'

def patch_cc1(sub_tuple):
    assert isinstance(sub_tuple, tuple)
    assert len(sub_tuple) == 2
    subst_name = sub_tuple[0]

    # Match `%clang_cc1`. The `ToolSubst` class inside `lit` adds a bunch of
    # regexes around this substitution so we try to match them here to avoid
    # matching a substitution that has `%clang_cc1` as a substring. This is
    # fragile
    if r'%\bclang_cc1\b' not in subst_name:
        return sub_tuple
    patched_sub = sub_tuple[1] + f' {extra_flags}'
    cc1_substituted[0] = True
    return (sub_tuple[0], patched_sub)

config.substitutions = list(map(patch_cc1, config.substitutions))

dir_with_patched_sub = os.path.dirname(__file__)
if not cc1_substituted[0]:
    lit_config.fatal(f'Failed to add `{extra_flags}` to %clang_cc1 invocations under {dir_with_patched_sub}')
else:
    lit_config.note(f'Implicitly passing `{extra_flags}` to %clang_cc1 invocations under {dir_with_patched_sub}')
