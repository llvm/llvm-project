/* -*- Mode: C++ -*- */

#ifndef _STACK_H
#define _STACK_H

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>

#include "debug_util.h"


// TB 20130123: I'm using my own custom stack type to let me
// performance engineer this later.
/*
 * Stack data structure for storing and maintaining data
 * associated with the call stack.
 */
template <typename STACK_DATA_T>
class Stack_t {
private:
  /* Default capacity for call stack.  Tunable to minimize
   * resizing. */
  static const uint32_t DEFAULT_CAPACITY = 128;

  /* call stack, implemented as an array of STACK_DATA_T's */
  STACK_DATA_T *_stack;
  /* current capacity of call stack */
  uint32_t _capacity;
  /* current head of call stack */
  uint32_t _head;

  /* General method to resize the call stack.
   * Called by _double_cap() and _halve_cap().
   *
   * @param new_capacity New capacity of the call stack.
   */
  void _resize(uint32_t new_capacity) {
    // Save a pointer to the call stack
    STACK_DATA_T *old_stack = _stack;
    // Allocate new call stack array
    _stack = new STACK_DATA_T[new_capacity];
    // Determine amount to copy over
    uint32_t copy_end = _capacity > new_capacity ? new_capacity : _capacity;

    // Copy contents of old call stack
    for (uint32_t i = 0; i < copy_end; ++i) {
      _stack[i] = old_stack[i];
    }
    _capacity = new_capacity;

    // Delete old call_stack
    delete[] old_stack;
  }

  /*
   * Doubles the capacity of the call stack.
   */
  void _double_cap() { _resize(_capacity * 2); }

  /*
   * Halves the capacity of the call stack.
   */
  void _halve_cap() { _resize(_capacity / 2); }


public:
  /*
   * Default constructor.
   */
  Stack_t() :
    _capacity(DEFAULT_CAPACITY),
    _head(0)
  { _stack = new STACK_DATA_T[_capacity]; }

  /*
   * Destructor.
   */
  ~Stack_t() { delete[] _stack; }

  /*
   * Simulate entering a function.  Effectively pushes a new
   * STACK_DATA_T onto the head of the call stack.
   */
  void push() {
    ++_head;

    if (_head == _capacity) {
      _double_cap();
    }
  }

  /*
   * Simulate exiting a function.  Effectively pops the head
   * STACK_DATA_T off of the stack.
   */
  void pop() {
    --_head;
    if (_capacity > DEFAULT_CAPACITY && _head < _capacity / 2) {
      _halve_cap();
    }
  }

  /*
   * Retrieves an arbitrary ancestor's STACK_DATA_T, specifically a
   * pointer to that data on the call stack.
   *
   * @param i the ancestor for the call at the head of the stack,
   * where i = 0 indicates the head of the call stack.
   */
  STACK_DATA_T* ancestor(uint32_t i) const {
    assert(i <= _head);
    cilksan_assert(_head < _capacity);
    return &(_stack[_head - i]);
  }

  /*
   * Retrieves a STACK_DATA_T at index i, specifically a
   * pointer to that data on the call stack.
   *
   * @param i the index of the stack element,
   * where element at index 0 is the oldest element.
   */
  STACK_DATA_T* at(uint32_t i) const {
    assert(i >= 0 && i <= _head);
    cilksan_assert(_head < _capacity);
    return &(_stack[i]);
  }

  /*
   * Retrieves the STACK_DATA_T at the head of the call stack.
   */
  STACK_DATA_T* head() const {
    return ancestor(0);
  }

  /*
   * Returns the current size of the stack, i.e. the number of entries
   * on the stack.
   */
  uint32_t size() const {
    return _head + 1;
  }

};

#endif // #define _STACK_H
