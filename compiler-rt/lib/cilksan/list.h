/* -*- Mode: C++ -*- */
/*
 * A list of pointers. It uses table-doubling to support larger capacities, but
 * does not release memory until explicitly told to do so (free()). It has a
 * crude form of locking in which a second lock before an unlock will crash the
 * program.
 *
 * Usage:
 *   - First lock the list via list.lock()
 *
 *   - Next, call list.push(obj) as many times as desired
 *
 *   - You can iterate over the list, e.g.
 *
 *      // Note that it's important to use the getter functions for the list
 *      // and length if you plan on pushing while iterating. Otherwise, it's
 *      // fine to cache the values.
 *      for (int i = 0; i < list.length(); i++) {
 *        void *obj = list.list()[i];
 *        // do something with obj
 *      }
 *
 *   - Finally, the list should eventually be freed using list.free_list()
 */

#ifndef _LIST_H
#define _LIST_H

#include <cstdlib>
#include "debug_util.h"

class List_t {
private:
  const int _DEFAULT_CAPACITY = 128;

  int _length = 0;
  void **_list = NULL;
#if CILKSAN_DEBUG
  bool _locked = false;
#endif
  int _capacity = 0;

public:
  List_t() :
      _length(0),
#if CILKSAN_DEBUG
      _locked(false),
#endif
      _capacity(_DEFAULT_CAPACITY) {

    _list = (void**)malloc(_capacity * sizeof(void*));
  }

  // Returns a pointer to the first element in the list. Elements are stored
  // contiguously.
  //
  // This must be called again after a push() in case the list has changed.
  //
  // The ordering of the elements will not be changed, even if the result
  // changes.
  __attribute__((always_inline))
  void **list() { return _list; }

  // The length of the list. Automically reset to 0 on unlock().
  __attribute__((always_inline))
  int length() { return _length; }

  // Crashes the program if lock() is called a second time before unlock().
  __attribute__((always_inline))
  void lock() {
    cilksan_assert(!_locked);

#if CILKSAN_DEBUG
    _locked = true;
#endif
  }

  __attribute__((always_inline))
  void unlock() {
    cilksan_assert(_locked);

#if CILKSAN_DEBUG
    _locked = false;
#endif
    _length = 0;
  }

  // Reclaims any memory used by the list. Should be called at the end of the
  // program.
  __attribute__((always_inline))
  void free_list() {
    cilksan_assert(!_locked);

    if (_list != NULL)
      free(_list);
  }

  // Adds an element to the end of the list.
  __attribute__((always_inline))
  void push(void *obj) {
    cilksan_assert(_locked);

    if (__builtin_expect(_length == _capacity, 0)) {
      _capacity *= 2;
      _list = (void**)realloc(_list, _capacity * sizeof(void*));
    }

    _list[_length++] = obj;
  }
};

#endif  // _LIST_H
