import lit.formats
import os

config.name = 'cfi' + config.name_suffix
config.suffixes = ['.c', '.cpp', '.test']
config.test_source_root = os.path.dirname(__file__)

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

clang = build_invocation([config.target_cflags])
clangxx = build_invocation([config.target_cflags] + config.cxx_mode_flags)

config.substitutions.append((r"%clang ", clang + ' '))
config.substitutions.append((r"%clangxx ", clangxx + ' '))

if 'darwin' in config.available_features:
  # -fsanitize=cfi is not supported on Darwin hosts
  config.unsupported = True
elif config.lto_supported:
  clang_cfi = clang + '-fsanitize=cfi '

  if config.cfi_lit_test_mode == "Devirt":
    config.available_features.add('devirt')
    clang_cfi += '-fwhole-program-vtables '
    config.substitutions.append((r"%expect_crash_unless_devirt ", ""))
  else:
    config.substitutions.append((r"%expect_crash_unless_devirt ", config.expect_crash))

  cxx = ' '.join(config.cxx_mode_flags) + ' '
  diag = '-fno-sanitize-trap=cfi -fsanitize-recover=cfi '
  non_dso = '-fvisibility=hidden '
  dso = '-fsanitize-cfi-cross-dso -fvisibility=default '
  if config.android:
    dso += '-include ' + config.test_source_root + '/cross-dso/util/cfi_stubs.h '
  config.substitutions.append((r"%clang_cfi ", clang_cfi + non_dso))
  config.substitutions.append((r"%clangxx_cfi ", clang_cfi + cxx + non_dso))
  config.substitutions.append((r"%clang_cfi_diag ", clang_cfi + non_dso + diag))
  config.substitutions.append((r"%clangxx_cfi_diag ", clang_cfi + cxx + non_dso + diag))
  config.substitutions.append((r"%clangxx_cfi_dso ", clang_cfi + cxx + dso))
  config.substitutions.append((r"%clangxx_cfi_dso_diag ", clang_cfi + cxx + dso + diag))
  config.substitutions.append((r"%debug_info_flags", ' '.join(config.debug_info_flags)))
else:
  config.unsupported = True

if config.default_sanitizer_opts:
  config.environment['UBSAN_OPTIONS'] = ':'.join(config.default_sanitizer_opts)

if lit_config.params.get('check_supported', None) and config.unsupported:
  raise BaseException("Tests unsupported")
