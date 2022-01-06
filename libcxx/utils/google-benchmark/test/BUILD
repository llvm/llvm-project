TEST_COPTS = [
    "-pedantic",
    "-pedantic-errors",
    "-std=c++11",
    "-Wall",
    "-Wextra",
    "-Wshadow",
    #    "-Wshorten-64-to-32",
    "-Wfloat-equal",
    "-fstrict-aliasing",
]

PER_SRC_COPTS = ({
    "cxx03_test.cc": ["-std=c++03"],
    # Some of the issues with DoNotOptimize only occur when optimization is enabled
    "donotoptimize_test.cc": ["-O3"],
})

TEST_ARGS = ["--benchmark_min_time=0.01"]

PER_SRC_TEST_ARGS = ({
    "user_counters_tabular_test.cc": ["--benchmark_counters_tabular=true"],
    "repetitions_test.cc": [" --benchmark_repetitions=3"],
})

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "output_test_helper",
    testonly = 1,
    srcs = ["output_test_helper.cc"],
    hdrs = ["output_test.h"],
    copts = TEST_COPTS,
    deps = [
        "//:benchmark",
        "//:benchmark_internal_headers",
    ],
)

[
    cc_test(
        name = test_src[:-len(".cc")],
        size = "small",
        srcs = [test_src],
        args = TEST_ARGS + PER_SRC_TEST_ARGS.get(test_src, []),
        copts = TEST_COPTS + PER_SRC_COPTS.get(test_src, []),
        deps = [
            ":output_test_helper",
            "//:benchmark",
            "//:benchmark_internal_headers",
            "@com_google_googletest//:gtest",
        ] + (
            ["@com_google_googletest//:gtest_main"] if (test_src[-len("gtest.cc"):] == "gtest.cc") else []
        ),
        # FIXME: Add support for assembly tests to bazel.
        # See Issue #556
        # https://github.com/google/benchmark/issues/556
    )
    for test_src in glob(
        ["*test.cc"],
        exclude = [
            "*_assembly_test.cc",
            "link_main_test.cc",
        ],
    )
]

cc_test(
    name = "link_main_test",
    size = "small",
    srcs = ["link_main_test.cc"],
    copts = TEST_COPTS,
    deps = ["//:benchmark_main"],
)
