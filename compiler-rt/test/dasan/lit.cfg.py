config.name = "DASAN" + config.name_suffix
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".c"]
config.excludes = ["Inputs"]

default_dasan_opts = "verbosity=1"
config.environment["DASAN_OPTIONS"] = default_dasan_opts
config.substitutions.append(
    ("%env_dasan_opts=", "env DASAN_OPTIONS=" + default_dasan_opts + ":")
)


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [config.target_cflags]
config.substitutions.append(("%clang ", build_invocation(target_cflags)))
config.substitutions.append(
    ("%clang_dasan ", build_invocation(target_cflags + ["-fsanitize=daddress"]))
)

if config.target_os != "Linux" or config.target_arch not in ["x86_64", "aarch64"]:
    config.unsupported = True

if "dasan" in config.gpu_runtimes:
    if "hip" in config.available_features:
        config.available_features.add("dasan-hip")
    if "openmp-offload" in config.available_features:
        config.available_features.add("dasan-openmp-offload")


def add_dasan_substitution(name, base):
    for pattern, replacement in config.substitutions:
        if pattern == base:
            config.substitutions.append(
                (name, replacement.rstrip() + " -fsanitize=daddress ")
            )
            return
    lit_config.fatal("Missing substitution %r" % base)


if "dasan-hip" in config.available_features:
    add_dasan_substitution("%clang_hip_dasan ", "%clang_hip ")
if "dasan-openmp-offload" in config.available_features:
    add_dasan_substitution("%clang_omp_offload_dasan ", "%clang_omp_offload ")

if config.dasan_global_support:
    config.available_features.add("dasan-globals")
