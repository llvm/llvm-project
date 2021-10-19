load("//bindings/python:build_defs.bzl", "py_extension")

py_library(
    name = "google_benchmark",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_benchmark",
        # pip; absl:app
    ],
)

py_extension(
    name = "_benchmark",
    srcs = ["benchmark.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//:benchmark",
        "@pybind11",
        "@python_headers",
    ],
)

py_test(
    name = "example",
    srcs = ["example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":google_benchmark",
    ],
)

