//===-- Spawn file actions  -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H
#define LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"
#include <spawn.h> // For mode_t

namespace LIBC_NAMESPACE_DECL {

struct BaseSpawnFileAction {
  enum ActionType {
    OPEN = 111,
    CLOSE = 222,
    DUP2 = 333,
  };

  ActionType type;
  BaseSpawnFileAction *next;

  /// Frees memory allocated for any of the actions. This is deliberately not
  /// the sized version, as SpawnFileOpenAction allocations are larger than the
  /// type.
  LIBC_INLINE static void operator delete(void *ptr) noexcept {
    ::operator delete(ptr);
  }

  /// Appends `act` to `actions`, which takes ownership of it.
  template <typename T>
  LIBC_INLINE static void add_action(posix_spawn_file_actions_t *actions,
                                     cpp::unique_ptr<T> act) {
    // posix_spawn_file_actions_destroy frees the actions through a
    // BaseSpawnFileAction pointer, without running their destructors.
    static_assert(cpp::is_trivially_destructible_v<T>,
                  "Spawn file actions must be trivially destructible");
    BaseSpawnFileAction *new_act = act.release();
    if (actions->__back != nullptr) {
      auto *back = reinterpret_cast<BaseSpawnFileAction *>(actions->__back);
      back->next = new_act;
      actions->__back = new_act;
    } else {
      // First action is being added.
      actions->__front = actions->__back = new_act;
    }
  }

protected:
  explicit BaseSpawnFileAction(ActionType t) : type(t), next(nullptr) {}
};

struct SpawnFileOpenAction : public BaseSpawnFileAction {
  int fd;
  int oflag;
  mode_t mode;
  // A copy of the path passed to posix_spawn_file_actions_addopen. It is
  // allocated together with the action (see create()).
  char path[];

  /// Allocates an open action, together with a copy of `p`. Returns a null
  /// CheckedPtr if the allocation fails.
  LIBC_INLINE static CheckedPtr<SpawnFileOpenAction>
  create(const char *p, int fdesc, int flags, mode_t m) {
    size_t path_size = internal::string_length(p) + 1;
    // As in make_unique, there is no need to inspect `ac`: operator new is
    // noexcept, so if it fails, no object is constructed and the
    // new-expression evaluates to nullptr.
    AllocChecker ac;
    return CheckedPtr<SpawnFileOpenAction>(
        new (ac, path_size) SpawnFileOpenAction(p, path_size, fdesc, flags, m));
  }

  /// Allocates storage for the object, followed by `path_size` bytes for the
  /// path. Declaring this also hides the global operator new, which would not
  /// leave room for the path.
  LIBC_INLINE static void *operator new(size_t size, AllocChecker &ac,
                                        size_t path_size) noexcept {
    return ::operator new(size + path_size, ac);
  }

private:
  LIBC_INLINE SpawnFileOpenAction(const char *p, size_t path_size, int fdesc,
                                  int flags, mode_t m)
      : BaseSpawnFileAction(BaseSpawnFileAction::OPEN), fd(fdesc), oflag(flags),
        mode(m) {
    inline_memcpy(path, p, path_size);
  }
};

struct SpawnFileCloseAction : public BaseSpawnFileAction {
  int fd;

  SpawnFileCloseAction(int fdesc)
      : BaseSpawnFileAction(BaseSpawnFileAction::CLOSE), fd(fdesc) {}
};

struct SpawnFileDup2Action : public BaseSpawnFileAction {
  int fd;
  int newfd;

  SpawnFileDup2Action(int fdesc, int new_fdesc)
      : BaseSpawnFileAction(BaseSpawnFileAction::DUP2), fd(fdesc),
        newfd(new_fdesc) {}
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H
