/*
   JavaScript-specific typemaps for LLDB
*/

%header %{
#include <v8.h>
%}

// Typemap for char ** (string arrays) - used in LaunchSimple, Launch, etc.
// Converts JavaScript arrays to C string arrays
%typemap(in) char ** {
  if ($input->IsArray()) {
    v8::Local<v8::Array> array = v8::Local<v8::Array>::Cast($input);
    uint32_t length = array->Length();
    $1 = (char **)malloc((length + 1) * sizeof(char *));

    for (uint32_t i = 0; i < length; i++) {
      v8::Local<v8::Value> element;
      if (array->Get(SWIGV8_CURRENT_CONTEXT(), i).ToLocal(&element)) {
        if (element->IsString()) {
          v8::String::Utf8Value str(SWIGV8_CURRENT_CONTEXT()->GetIsolate(), element);
          $1[i] = strdup(*str);
        } else {
          free($1);
          SWIG_exception_fail(SWIG_TypeError, "Array elements must be strings");
        }
      }
    }
    $1[length] = NULL;
  } else if ($input->IsNull() || $input->IsUndefined()) {
    $1 = NULL;
  } else {
    SWIG_exception_fail(SWIG_TypeError, "Expected array of strings or null");
  }
}

%typemap(freearg) char ** {
  if ($1) {
    for (int i = 0; $1[i] != NULL; i++) {
      free($1[i]);
    }
    free($1);
  }
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_STRING_ARRAY) char ** {
  $1 = $input->IsArray() || $input->IsNull() || $input->IsUndefined();
}
