//===- DiagnosticCASKinds.td - CAS diagnostics ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Component = "CAS" in {

def err_cas_cannot_be_initialized : Error<
  "CAS cannot be initialized from the specified '-fcas-*' options: %0">,
  DefaultFatal;
def err_cas_cannot_parse_root_id
    : Error<"CAS cannot parse root-id '%0' specified by '-fcas-*' options">,
      DefaultFatal;
def err_cas_filesystem_cannot_be_initialized : Error<
  "CAS filesystem cannot be initialized from root-id '%0': %1">, DefaultFatal;
def err_cas_filesystem_cannot_set_working_directory : Error<
  "CAS filesystem cannot set working directory to '%0' specified by"
  " -fcas-fs-working-directory">, DefaultFatal;
def err_cas_depscan_daemon_connection: Error<
  "failed to establish connection with depscan daemon: %0">, DefaultFatal;
def err_cas_depscan_failed: Error<
  "CAS-based dependency scan failed: %0">, DefaultFatal;
def err_cas_store: Error<"failed to store to CAS: %0">, DefaultFatal;
def err_cas_unloadable_module
    : Error<"module file '%0' not found: unloadable %select{casid|module cache "
            "key}1 %2: %3">,
      DefaultFatal;
def err_cas_missing_module : Error<
  "module file '%0' not found: missing module cache key %1: %2">, DefaultFatal;
def err_cas_missing_root_id : Error<
  "CAS missing expected root-id '%0'">, DefaultFatal;
def err_cas_cannot_parse_include_tree_id : Error<
  "CAS cannot parse include-tree-id '%0'">, DefaultFatal;
def err_cas_missing_include_tree_id : Error<
  "CAS missing expected include-tree '%0'">, DefaultFatal;
def err_cas_cannot_parse_input_cache_key : Error<
  "CAS cannot parse input cache key '%0'">, DefaultFatal;
def err_cas_cannot_lookup_input_cache_key : Error<
  "CAS cannot lookup input cache key '%0'">, DefaultFatal;
def err_cas_missing_input_cache_entry : Error<
  "CAS missing expected input cache key '%0'">, DefaultFatal;
def err_cas_cannot_load_api_notes_include_tree : Error<
  "cannot load APINotes from include-tree-id '%0'">, DefaultFatal;

def warn_clang_cache_disabled_caching: Warning<
  "caching disabled because %0">,
  InGroup<DiagGroup<"clang-cache">>;
def err_clang_cache_failed_execution: Error<
  "clang-cache failed to execute compiler: %0">;
def err_clang_cache_cannot_find_binary: Error<
  "clang-cache cannot find compiler binary %0">;
def err_clang_cache_missing_compiler_command: Error<
  "missing compiler command for clang-cache">;
def err_clang_cache_scanserve_missing_args: Error<
  "missing arguments for dep-scanning server mode">;
def err_caching_backend_fail: Error<
  "caching backend error: %0">, DefaultFatal;
def err_caching_output_non_deterministic: Error<
  "caching failed because the output can be non-deterministic">;

def remark_compile_job_cache_hit : Remark<
  "compile job cache hit for '%0' => '%1'">, InGroup<CompileJobCacheHit>;
def remark_compile_job_cache_miss : Remark<
  "compile job cache miss for '%0'">, InGroup<CompileJobCacheMiss>;
def remark_compile_job_cache_miss_result_not_found : Remark<
  "compile job cache miss for '%0' (result not found: '%1')">,
  InGroup<CompileJobCacheMiss>;
def remark_compile_job_cache_backend_output_not_found : Remark<
  "compile job cache backend did not find output '%0' for key '%1' (output ID: '%2')">,
  InGroup<CompileJobCacheMiss>;
def remark_compile_job_cache_skipped : Remark<
  "compile job cache skipped for '%0'">, InGroup<CompileJobCache>;
def remark_compile_job_cache_timing_depscan : Remark<
  "compile job dependency scanning time: %0">, InGroup<CompileJobCacheTiming>;
def remark_compile_job_cache_timing_backend_key_query : Remark<
  "compile job cache backend key query time: %0">, InGroup<CompileJobCacheTiming>;
def remark_compile_job_cache_timing_backend_key_update : Remark<
  "compile job cache backend key update time: %0">, InGroup<CompileJobCacheTiming>;
def remark_compile_job_cache_timing_backend_load : Remark<
  "compile job cache backend load artifacts time: %0">, InGroup<CompileJobCacheTiming>;
def remark_compile_job_cache_timing_backend_store : Remark<
  "compile job cache backend store artifacts time: %0">, InGroup<CompileJobCacheTiming>;

} // let Component = "CAS" in
