//===- DiagnosticCASKinds.td - CAS diagnostics ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Component = "CAS" in {

def err_builtin_cas_cannot_be_initialized : Error<
  "CAS cannot be initialized from '%0' on disk (check -fcas-path)">,
  DefaultFatal;
def err_cas_cannot_parse_root_id : Error<
  "CAS cannot parse root-id '%0' specified by -fcas-fs">, DefaultFatal;
def err_cas_filesystem_cannot_be_initialized : Error<
  "CAS filesystem cannot be initialized from root-id '%0' specified by"
  " -fcas-fs">, DefaultFatal;
def err_cas_filesystem_cannot_set_working_directory : Error<
  "CAS filesystem cannot set working directory to '%0' specified by"
  " -fcas-fs-working-directory">, DefaultFatal;
def err_cas_depscan_daemon_connection: Error<
  "Failed to establish connection with depscan daemon: %0">, DefaultFatal;
def err_cas_depscan_failed: Error<
  "CAS-based dependency scan failed: %0">, DefaultFatal;

def warn_clang_cache_disabled_caching: Warning<
  "clang-cache invokes a different clang binary than itself, it will perform "
  "a normal non-caching invocation of the compiler">,
  InGroup<DiagGroup<"clang-cache">>;
def err_clang_cache_failed_execution: Error<
  "clang-cache failed to execute compiler: %0">;
def err_clang_cache_cannot_find_binary: Error<
  "clang-cache cannot find compiler binary %0">;
def err_clang_cache_missing_compiler_command: Error<
  "missing compiler command for clang-cache">;

def remark_compile_job_cache_hit : Remark<
  "compile job cache hit for '%0' => '%1'">, InGroup<CompileJobCacheHit>;
def remark_compile_job_cache_miss : Remark<
  "compile job cache miss for '%0'">, InGroup<CompileJobCacheMiss>;
def remark_compile_job_cache_miss_result_not_found : Remark<
  "compile job cache miss for '%0' (result not found: '%1')">,
  InGroup<CompileJobCacheMiss>;

} // let Component = "CAS" in
