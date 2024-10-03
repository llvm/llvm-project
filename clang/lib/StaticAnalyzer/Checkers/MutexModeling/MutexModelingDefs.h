//===--- MutexModelingDefs.h - Modeling of mutexes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the default set of events that are handled by the mutex modeling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGMUTEXMODELINGDEFS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGMUTEXMODELINGDEFS_H

#include "MutexModelingDomain.h"
#include "MutexRegionExtractor.h"

#include <vector>

namespace clang::ento::mutex_modeling {

static auto getHandledEvents(){return std::vector<EventDescriptor> {
  // - Pthread
  EventDescriptor{MakeFirstArgExtractor({"pthread_mutex_init"}), EventKind::Init,
                  LibraryKind::Pthread},
#if 0
             // TODO: pthread_rwlock_init(2 arguments).
             // TODO: lck_mtx_init(3 arguments).
             // TODO: lck_mtx_alloc_init(2 arguments) => returns the mutex.
             // TODO: lck_rw_init(3 arguments).
             // TODO: lck_rw_alloc_init(2 arguments) => returns the mutex.

             // - Fuchsia
             EventDescriptor{FirstArgMutexExtractor{CallDescription{
                                 CDM::CLibrary, {"spin_lock_init"}, 1}},
                             EventKind::Init, LibraryKind::Fuchsia},

             // - C11
             EventDescriptor{FirstArgMutexExtractor{CallDescription{
                                 CDM::CLibrary, {"mtx_init"}, 2}},
                             EventKind::Init, LibraryKind::C11},

             // Acquire kind
             // - Pthread
             //
             EventDescriptor{FirstArgMutexExtractor{CallDescription{
                                 CDM::CLibrary, {"pthread_mutex_lock"}, 1}},
                             Event::Acquire, LibraryKind::Pthread,
                             SemanticsKind::PthreadSemantics},
             EventDescriptor{FirstArgMutexExtractor{
                 CallDescription{CDM::CLibrary, {"pthread_rwlock_rdlock"}, 1},
                 Event::Acquire, LibraryKind::Pthread,
                 SemanticsKind::PthreadSemantics}},
             EventDescriptor{FirstArgMutexExtractor{CallDescription{
                                 CDM::CLibrary, {"pthread_rwlock_wrlock"}, 1}},
                             Event::Acquire, LibraryKind::Pthread,
                             SemanticsKind::PthreadSemantics}},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_rwlock_wrlock"}, 1},
             Event::Acquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{CallDescription{CDM::CLibrary, {"lck_mtx_lock"}, 1},
                         Event::Acquire, Syntax::FirstArg,
                         LockingSemantics::XNUSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_lock_exclusive"}, 1},
             Event::Acquire, Syntax::FirstArg, LockingSemantics::XNUSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_lock_shared"}, 1},
             Event::Acquire, Syntax::FirstArg, LockingSemantics::XNUSemantics},

         // - Fuchsia
         EventDescriptor{CallDescription{CDM::CLibrary, {"spin_lock"}, 1},
                         Event::Acquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},
         EventDescriptor{CallDescription{CDM::CLibrary, {"spin_lock_save"}, 3},
                         Event::Acquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},
         EventDescriptor{CallDescription{CDM::CLibrary, {"sync_mutex_lock"}, 1},
                         Event::Acquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"sync_mutex_lock_with_waiter"}, 1},
             Event::Acquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},

         // - C11
         EventDescriptor{CallDescription{CDM::CLibrary, {"mtx_lock"}, 1},
                         Event::Acquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},

         // - std
         EventDescriptor{
             CallDescription{CDM::CXXMethod, {"std", "mutex", "lock"}, 0},
             Event::Acquire, Syntax::Member,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CXXMethod, {"std", "lock_guard"}, 1},
             Event::Acquire, Syntax::RAII, LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CXXMethod, {"std", "unique_lock"}, 1},
             Event::Acquire, Syntax::RAII, LockingSemantics::PthreadSemantics},

         // TryAcquire kind
         // - Pthread
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_mutex_trylock"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_rwlock_tryrdlock"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_rwlock_trywrlock"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_mtx_try_lock"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::XNUSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_try_lock_exclusive"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::XNUSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_try_lock_shared"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::XNUSemantics},

         // - Fuchsia
         EventDescriptor{CallDescription{CDM::CLibrary, {"spin_trylock"}, 1},
                         Event::TryAcquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"sync_mutex_trylock"}, 1},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"sync_mutex_timedlock"}, 2},
             Event::TryAcquire, Syntax::FirstArg,
             LockingSemantics::PthreadSemantics},

         // - C11
         EventDescriptor{CallDescription{CDM::CLibrary, {"mtx_trylock"}, 1},
                         Event::TryAcquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},
         EventDescriptor{CallDescription{CDM::CLibrary, {"mtx_timedlock"}, 2},
                         Event::TryAcquire, Syntax::FirstArg,
                         LockingSemantics::PthreadSemantics},

         // Release kind
         // - Pthread
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_mutex_unlock"}, 1},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"pthread_rwlock_unlock"}, 1},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},
         EventDescriptor{CallDescription{CDM::CLibrary, {"lck_mtx_unlock"}, 1},
                         Event::Release, Syntax::FirstArg,
                         LockingSemantics::NotApplicable},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_unlock_exclusive"}, 1},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"lck_rw_unlock_shared"}, 1},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},
         EventDescriptor{CallDescription{CDM::CLibrary, {"lck_rw_done"}, 1},
                         Event::Release, Syntax::FirstArg,
                         LockingSemantics::NotApplicable},

         // - Fuchsia
         EventDescriptor{CallDescription{CDM::CLibrary, {"spin_unlock"}, 1},
                         Event::Release, Syntax::FirstArg,
                         LockingSemantics::NotApplicable},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"spin_unlock_restore"}, 3},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},
         EventDescriptor{
             CallDescription{CDM::CLibrary, {"sync_mutex_unlock"}, 1},
             Event::Release, Syntax::FirstArg, LockingSemantics::NotApplicable},

         // - C11
         EventDescriptor{CallDescription{CDM::CLibrary, {"mtx_unlock"}, 1},
                         Event::Release, Syntax::FirstArg,
                         LockingSemantics::NotApplicable},

         // - std
         EventDescriptor{
             CallDescription{CDM::CXXMethod, {"std", "mutex", "unlock"}, 0},
             Event::Release, Syntax::Member, LockingSemantics::NotApplicable},

         // Destroy kind
         // - Pthread
         EventDescriptor{{CDM::CLibrary, {"pthread_mutex_destroy"}, 1},
                         Event::Destroy,
                         Syntax::FirstArg,
                         LockingSemantics::NotApplicable},
         EventDescriptor{CallDescription{CDM::CLibrary, {"lck_mtx_destroy"}, 2},
                         Event::Destroy, Syntax::FirstArg,
                         LockingSemantics::NotApplicable},
         // TODO: pthread_rwlock_destroy(1 argument).
         // TODO: lck_rw_destroy(2 arguments).

         // - C11
         EventDescriptor{CallDescription{CDM::CLibrary, {"mtx_destroy"}, 1},
                         Event::Destroy, Syntax::FirstArg,
                         LockingSemantics::NotApplicable}
#endif
};
} // namespace clang::ento::mutex_modeling

} // namespace clang::ento::mutex_modeling

#endif
