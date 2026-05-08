// RUN: %check_clang_tidy -check-suffixes=THRESHOLD %s bugprone-suspicious-missing-comma %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-suspicious-missing-comma.SizeThreshold: 3, \
// RUN:     bugprone-suspicious-missing-comma.RatioThreshold: '.5', \
// RUN:     bugprone-suspicious-missing-comma.MaxConcatenatedTokens: 3 \
// RUN:   }}"

const char *SmallArray[] = {
    "hello" "world",
    "foo",
    "bar",
};
// CHECK-MESSAGES-THRESHOLD: :[[@LINE-4]]:5: warning: suspicious string literal, probably missing a comma [bugprone-suspicious-missing-comma]

const char *ManyTokensArray[] = {
    "a" "b" "c",
    "d",
    "e",
};

const char *TwoTokensArray[] = {
    "a" "b",
    "c",
    "d",
};
// CHECK-MESSAGES-THRESHOLD: :[[@LINE-4]]:5: warning: suspicious string literal, probably missing a comma [bugprone-suspicious-missing-comma]

const char *HighRatioArray[] = {
    "a" "b",
    "c" "d",
    "e",
};
