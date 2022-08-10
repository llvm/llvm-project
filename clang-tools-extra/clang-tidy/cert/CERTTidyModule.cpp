//===--- CERTTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../bugprone/BadSignalToKillThreadCheck.h"
#include "../bugprone/ReservedIdentifierCheck.h"
#include "../bugprone/SignalHandlerCheck.h"
#include "../bugprone/SignedCharMisuseCheck.h"
#include "../bugprone/SpuriouslyWakeUpFunctionsCheck.h"
#include "../bugprone/SuspiciousMemoryComparisonCheck.h"
#include "../bugprone/UnhandledSelfAssignmentCheck.h"
#include "../bugprone/UnusedReturnValueCheck.h"
#include "../concurrency/ThreadCanceltypeAsynchronousCheck.h"
#include "../google/UnnamedNamespaceInHeaderCheck.h"
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/NonCopyableObjects.h"
#include "../misc/StaticAssertCheck.h"
#include "../misc/ThrowByValueCatchByReferenceCheck.h"
#include "../performance/MoveConstructorInitCheck.h"
#include "../readability/UppercaseLiteralSuffixCheck.h"
#include "CommandProcessorCheck.h"
#include "DefaultOperatorNewAlignmentCheck.h"
#include "DontModifyStdNamespaceCheck.h"
#include "FloatLoopCounter.h"
#include "LimitedRandomnessCheck.h"
#include "MutatingCopyCheck.h"
#include "NonTrivialTypesLibcMemoryCallsCheck.h"
#include "PostfixOperatorCheck.h"
#include "ProperlySeededRandomGeneratorCheck.h"
#include "SetLongJmpCheck.h"
#include "StaticObjectExceptionCheck.h"
#include "StrToNumCheck.h"
#include "ThrownExceptionTypeCheck.h"
#include "VariadicFunctionDefCheck.h"

namespace {

// Checked functions for cert-err33-c.
// The following functions are deliberately excluded because they can be called
// with NULL argument and in this case the check is not applicable:
// `mblen, mbrlen, mbrtowc, mbtowc, wctomb, wctomb_s`.
// FIXME: The check can be improved to handle such cases.
const llvm::StringRef CertErr33CCheckedFunctions = "::aligned_alloc;"
                                                   "::asctime_s;"
                                                   "::at_quick_exit;"
                                                   "::atexit;"
                                                   "::bsearch;"
                                                   "::bsearch_s;"
                                                   "::btowc;"
                                                   "::c16rtomb;"
                                                   "::c32rtomb;"
                                                   "::calloc;"
                                                   "::clock;"
                                                   "::cnd_broadcast;"
                                                   "::cnd_init;"
                                                   "::cnd_signal;"
                                                   "::cnd_timedwait;"
                                                   "::cnd_wait;"
                                                   "::ctime_s;"
                                                   "::fclose;"
                                                   "::fflush;"
                                                   "::fgetc;"
                                                   "::fgetpos;"
                                                   "::fgets;"
                                                   "::fgetwc;"
                                                   "::fopen;"
                                                   "::fopen_s;"
                                                   "::fprintf;"
                                                   "::fprintf_s;"
                                                   "::fputc;"
                                                   "::fputs;"
                                                   "::fputwc;"
                                                   "::fputws;"
                                                   "::fread;"
                                                   "::freopen;"
                                                   "::freopen_s;"
                                                   "::fscanf;"
                                                   "::fscanf_s;"
                                                   "::fseek;"
                                                   "::fsetpos;"
                                                   "::ftell;"
                                                   "::fwprintf;"
                                                   "::fwprintf_s;"
                                                   "::fwrite;"
                                                   "::fwscanf;"
                                                   "::fwscanf_s;"
                                                   "::getc;"
                                                   "::getchar;"
                                                   "::getenv;"
                                                   "::getenv_s;"
                                                   "::gets_s;"
                                                   "::getwc;"
                                                   "::getwchar;"
                                                   "::gmtime;"
                                                   "::gmtime_s;"
                                                   "::localtime;"
                                                   "::localtime_s;"
                                                   "::malloc;"
                                                   "::mbrtoc16;"
                                                   "::mbrtoc32;"
                                                   "::mbsrtowcs;"
                                                   "::mbsrtowcs_s;"
                                                   "::mbstowcs;"
                                                   "::mbstowcs_s;"
                                                   "::memchr;"
                                                   "::mktime;"
                                                   "::mtx_init;"
                                                   "::mtx_lock;"
                                                   "::mtx_timedlock;"
                                                   "::mtx_trylock;"
                                                   "::mtx_unlock;"
                                                   "::printf_s;"
                                                   "::putc;"
                                                   "::putwc;"
                                                   "::raise;"
                                                   "::realloc;"
                                                   "::remove;"
                                                   "::rename;"
                                                   "::scanf;"
                                                   "::scanf_s;"
                                                   "::setlocale;"
                                                   "::setvbuf;"
                                                   "::signal;"
                                                   "::snprintf;"
                                                   "::snprintf_s;"
                                                   "::sprintf;"
                                                   "::sprintf_s;"
                                                   "::sscanf;"
                                                   "::sscanf_s;"
                                                   "::strchr;"
                                                   "::strerror_s;"
                                                   "::strftime;"
                                                   "::strpbrk;"
                                                   "::strrchr;"
                                                   "::strstr;"
                                                   "::strtod;"
                                                   "::strtof;"
                                                   "::strtoimax;"
                                                   "::strtok;"
                                                   "::strtok_s;"
                                                   "::strtol;"
                                                   "::strtold;"
                                                   "::strtoll;"
                                                   "::strtoul;"
                                                   "::strtoull;"
                                                   "::strtoumax;"
                                                   "::strxfrm;"
                                                   "::swprintf;"
                                                   "::swprintf_s;"
                                                   "::swscanf;"
                                                   "::swscanf_s;"
                                                   "::thrd_create;"
                                                   "::thrd_detach;"
                                                   "::thrd_join;"
                                                   "::thrd_sleep;"
                                                   "::time;"
                                                   "::timespec_get;"
                                                   "::tmpfile;"
                                                   "::tmpfile_s;"
                                                   "::tmpnam;"
                                                   "::tmpnam_s;"
                                                   "::tss_create;"
                                                   "::tss_get;"
                                                   "::tss_set;"
                                                   "::ungetc;"
                                                   "::ungetwc;"
                                                   "::vfprintf;"
                                                   "::vfprintf_s;"
                                                   "::vfscanf;"
                                                   "::vfscanf_s;"
                                                   "::vfwprintf;"
                                                   "::vfwprintf_s;"
                                                   "::vfwscanf;"
                                                   "::vfwscanf_s;"
                                                   "::vprintf_s;"
                                                   "::vscanf;"
                                                   "::vscanf_s;"
                                                   "::vsnprintf;"
                                                   "::vsnprintf_s;"
                                                   "::vsprintf;"
                                                   "::vsprintf_s;"
                                                   "::vsscanf;"
                                                   "::vsscanf_s;"
                                                   "::vswprintf;"
                                                   "::vswprintf_s;"
                                                   "::vswscanf;"
                                                   "::vswscanf_s;"
                                                   "::vwprintf_s;"
                                                   "::vwscanf;"
                                                   "::vwscanf_s;"
                                                   "::wcrtomb;"
                                                   "::wcschr;"
                                                   "::wcsftime;"
                                                   "::wcspbrk;"
                                                   "::wcsrchr;"
                                                   "::wcsrtombs;"
                                                   "::wcsrtombs_s;"
                                                   "::wcsstr;"
                                                   "::wcstod;"
                                                   "::wcstof;"
                                                   "::wcstoimax;"
                                                   "::wcstok;"
                                                   "::wcstok_s;"
                                                   "::wcstol;"
                                                   "::wcstold;"
                                                   "::wcstoll;"
                                                   "::wcstombs;"
                                                   "::wcstombs_s;"
                                                   "::wcstoul;"
                                                   "::wcstoull;"
                                                   "::wcstoumax;"
                                                   "::wcsxfrm;"
                                                   "::wctob;"
                                                   "::wctrans;"
                                                   "::wctype;"
                                                   "::wmemchr;"
                                                   "::wprintf_s;"
                                                   "::wscanf;"
                                                   "::wscanf_s;";

} // namespace

namespace clang {
namespace tidy {
namespace cert {

class CERTModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    // C++ checkers
    // CON
    CheckFactories.registerCheck<bugprone::SpuriouslyWakeUpFunctionsCheck>(
        "cert-con54-cpp");
    // DCL
    CheckFactories.registerCheck<PostfixOperatorCheck>(
        "cert-dcl21-cpp");
    CheckFactories.registerCheck<VariadicFunctionDefCheck>("cert-dcl50-cpp");
    CheckFactories.registerCheck<bugprone::ReservedIdentifierCheck>(
        "cert-dcl51-cpp");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "cert-dcl54-cpp");
    CheckFactories.registerCheck<DontModifyStdNamespaceCheck>(
        "cert-dcl58-cpp");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "cert-dcl59-cpp");
    // ERR
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err09-cpp");
    CheckFactories.registerCheck<SetLongJmpCheck>("cert-err52-cpp");
    CheckFactories.registerCheck<StaticObjectExceptionCheck>("cert-err58-cpp");
    CheckFactories.registerCheck<ThrownExceptionTypeCheck>("cert-err60-cpp");
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err61-cpp");
    // MEM
    CheckFactories.registerCheck<DefaultOperatorNewAlignmentCheck>(
        "cert-mem57-cpp");
    // MSC
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc50-cpp");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc51-cpp");
    CheckFactories.registerCheck<bugprone::SignalHandlerCheck>(
        "cert-msc54-cpp");
    // OOP
    CheckFactories.registerCheck<performance::MoveConstructorInitCheck>(
        "cert-oop11-cpp");
    CheckFactories.registerCheck<bugprone::UnhandledSelfAssignmentCheck>(
        "cert-oop54-cpp");
    CheckFactories.registerCheck<NonTrivialTypesLibcMemoryCallsCheck>(
        "cert-oop57-cpp");
    CheckFactories.registerCheck<MutatingCopyCheck>(
        "cert-oop58-cpp");

    // C checkers
    // CON
    CheckFactories.registerCheck<bugprone::SpuriouslyWakeUpFunctionsCheck>(
        "cert-con36-c");
    // DCL
    CheckFactories.registerCheck<misc::StaticAssertCheck>("cert-dcl03-c");
    CheckFactories.registerCheck<readability::UppercaseLiteralSuffixCheck>(
        "cert-dcl16-c");
    CheckFactories.registerCheck<bugprone::ReservedIdentifierCheck>(
        "cert-dcl37-c");
    // ENV
    CheckFactories.registerCheck<CommandProcessorCheck>("cert-env33-c");
    // ERR
    CheckFactories.registerCheck<bugprone::UnusedReturnValueCheck>(
        "cert-err33-c");
    CheckFactories.registerCheck<StrToNumCheck>("cert-err34-c");
    // EXP
    CheckFactories.registerCheck<bugprone::SuspiciousMemoryComparisonCheck>(
        "cert-exp42-c");
    // FLP
    CheckFactories.registerCheck<FloatLoopCounter>("cert-flp30-c");
    CheckFactories.registerCheck<bugprone::SuspiciousMemoryComparisonCheck>(
        "cert-flp37-c");
    // FIO
    CheckFactories.registerCheck<misc::NonCopyableObjectsCheck>("cert-fio38-c");
    // MSC
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc30-c");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc32-c");
    // POS
    CheckFactories.registerCheck<bugprone::BadSignalToKillThreadCheck>(
        "cert-pos44-c");
    CheckFactories
        .registerCheck<concurrency::ThreadCanceltypeAsynchronousCheck>(
            "cert-pos47-c");
    // SIG
    CheckFactories.registerCheck<bugprone::SignalHandlerCheck>("cert-sig30-c");
    // STR
    CheckFactories.registerCheck<bugprone::SignedCharMisuseCheck>(
        "cert-str34-c");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    ClangTidyOptions::OptionMap &Opts = Options.CheckOptions;
    Opts["cert-dcl16-c.NewSuffixes"] = "L;LL;LU;LLU";
    Opts["cert-err33-c.CheckedFunctions"] = CertErr33CCheckedFunctions;
    Opts["cert-oop54-cpp.WarnOnlyIfThisHasSuspiciousField"] = "false";
    Opts["cert-str34-c.DiagnoseSignedUnsignedCharComparisons"] = "false";
    return Options;
  }
};

} // namespace cert

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<cert::CERTModule>
    X("cert-module",
      "Adds lint checks corresponding to CERT secure coding guidelines.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the CERTModule.
volatile int CERTModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
