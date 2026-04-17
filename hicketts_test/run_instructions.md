# Run Instructions

## Test the PR #144313 fix (makeValue/makeValueInplace)
```bash
clang-tidy -checks='bugprone-unchecked-optional-access' \
  test_makevalue_fix.cpp -- \
  -I ../clang-tools-extra/test/clang-tidy/checkers/bugprone/Inputs/unchecked-optional-access \
  -Wno-undefined-inline
```

## Test HickettsOptional (custom optional-like type)
```bash
clang-tidy -checks='bugprone-unchecked-optional-access' \
  test_hicketts_optional.cpp -- \
  -I . \
  -Wno-undefined-inline
```

All commands assume you are running from the `hicketts_test/` directory.
