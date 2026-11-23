// RUN: %clang_analyze_cc1 -verify %s

// expected-no-diagnostics

struct SizedBufferPtr {
  union {
    int* buffer_ptr;
  } /*anonymous*/;
  int size;
};

extern int global_buf_size;
extern int global_buf[];

void analysis_entry_point_rdar167136849(void) {
  // When binding the initial value for "buffers".
  // We will first bind the first array element: buffers[0] <- cv{cv{&Elem{global_buf, 0}}}
  // This is of type "struct SizedBufferPtr" so we dispatch this bind to bind by fields.
  // So we bind: buffers[0].union anonymous <- cv{&Elem{global_buf, 0}}
  // Because that is "isUnionType", its bound using "bindAggregate", thus add a default binding.
  // After this we have in the Store: { default at 0: cv{&Elem{global_buf, 0}} }
  // We then bind the second field: buffers[0].size <- reg{global_buf_size}
  // After this we have in the Store: { default at 0: cv{&Elem{global_buf, 0}}, direct at 64: reg{global_buf_size} }
  // We are done with the initializer expression, so we realize that
  // the second element of "buffers" didn't have an explicit initializer,
  // so we need to bind a suitable default value using "setImplicitDefaultValue".
  // That happens to add a default binding 0 to "buffer", which overwrites the previous default binding that "bindAggregate" added.
  // After this we falsely conclude that "buffers[0].buffer_ptr" is a nullptr.

  struct SizedBufferPtr buffers[2] = {
      {.buffer_ptr = global_buf, .size = global_buf_size},
      // [1] is not explicitly initialized, so it will be defaulted to zero.
  };
  *buffers[0].buffer_ptr = 2; // no-warning: writing indirectly to "global_buf" is fine
}
