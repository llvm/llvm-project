//===-- ReproducerInstrumentation.h -----------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H
#define LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <iostream>
#include <map>
#include <type_traits>

template <typename T,
          typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
inline void stringify_append(llvm::raw_string_ostream &ss, const T &t) {
  ss << t;
}

template <typename T, typename std::enable_if<!std::is_fundamental<T>::value,
                                              int>::type = 0>
inline void stringify_append(llvm::raw_string_ostream &ss, const T &t) {
  ss << &t;
}

template <typename T>
inline void stringify_append(llvm::raw_string_ostream &ss, T *t) {
  ss << reinterpret_cast<void *>(t);
}

template <typename T>
inline void stringify_append(llvm::raw_string_ostream &ss, const T *t) {
  ss << reinterpret_cast<const void *>(t);
}

template <>
inline void stringify_append<char>(llvm::raw_string_ostream &ss,
                                   const char *t) {
  ss << '\"' << t << '\"';
}

template <>
inline void stringify_append<std::nullptr_t>(llvm::raw_string_ostream &ss,
                                             const std::nullptr_t &t) {
  ss << "\"nullptr\"";
}

template <typename Head>
inline void stringify_helper(llvm::raw_string_ostream &ss, const Head &head) {
  stringify_append(ss, head);
}

template <typename Head, typename... Tail>
inline void stringify_helper(llvm::raw_string_ostream &ss, const Head &head,
                             const Tail &... tail) {
  stringify_append(ss, head);
  ss << ", ";
  stringify_helper(ss, tail...);
}

template <typename... Ts> inline std::string stringify_args(const Ts &... ts) {
  std::string buffer;
  llvm::raw_string_ostream ss(buffer);
  stringify_helper(ss, ts...);
  return ss.str();
}

// Define LLDB_REPRO_INSTR_TRACE to trace to stderr instead of LLDB's log
// infrastructure. This is useful when you need to see traces before the logger
// is initialized or enabled.
// #define LLDB_REPRO_INSTR_TRACE

#define LLDB_REGISTER_CONSTRUCTOR(Class, Signature)                            \
  R.Register<Class * Signature>(&construct<Class Signature>::doit, "", #Class, \
                                #Class, #Signature)

#define LLDB_REGISTER_METHOD(Result, Class, Method, Signature)                 \
  R.Register(                                                                  \
      &invoke<Result(Class::*) Signature>::method<(&Class::Method)>::doit,     \
      #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_METHOD_CONST(Result, Class, Method, Signature)           \
  R.Register(&invoke<Result(Class::*)                                          \
                         Signature const>::method<(&Class::Method)>::doit,     \
             #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_STATIC_METHOD(Result, Class, Method, Signature)          \
  R.Register(&invoke<Result(*) Signature>::method<(&Class::Method)>::doit,     \
             #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_CHAR_PTR_REDIRECT_STATIC(Result, Class, Method)          \
  R.Register(                                                                  \
      &invoke<Result (*)(char *, size_t)>::method<(&Class::Method)>::doit,     \
      &char_ptr_redirect<Result (*)(char *,                                    \
                                    size_t)>::method<(&Class::Method)>::doit,  \
      #Result, #Class, #Method, "(char*, size_t");

#define LLDB_REGISTER_CHAR_PTR_REDIRECT(Result, Class, Method)                 \
  R.Register(&invoke<Result (Class::*)(char *, size_t)>::method<(              \
                 &Class::Method)>::doit,                                       \
             &char_ptr_redirect<Result (Class::*)(char *, size_t)>::method<(   \
                 &Class::Method)>::doit,                                       \
             #Result, #Class, #Method, "(char*, size_t");

#define LLDB_REGISTER_CHAR_PTR_REDIRECT_CONST(Result, Class, Method)           \
  R.Register(&invoke<Result (Class::*)(char *, size_t)                         \
                         const>::method<(&Class::Method)>::doit,               \
             &char_ptr_redirect<Result (Class::*)(char *, size_t)              \
                                    const>::method<(&Class::Method)>::doit,    \
             #Result, #Class, #Method, "(char*, size_t");

#define LLDB_CONSTRUCT_(T, ...)                                                \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(this, __VA_ARGS__));  \
  if (lldb_private::repro::InstrumentationData _data =                         \
          LLDB_GET_INSTRUMENTATION_DATA()) {                                   \
    _recorder.Record(_data.GetSerializer(), _data.GetRegistry(),               \
                     &lldb_private::repro::construct<T>::doit, __VA_ARGS__);   \
    _recorder.RecordResult(this, false);                                       \
  }

#define LLDB_RECORD_CONSTRUCTOR(Class, Signature, ...)                         \
  LLDB_CONSTRUCT_(Class Signature, __VA_ARGS__)

#define LLDB_RECORD_CONSTRUCTOR_NO_ARGS(Class)                                 \
  LLDB_CONSTRUCT_(Class(), lldb_private::repro::EmptyArg())

#define LLDB_RECORD_(T1, T2, ...)                                              \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));        \
  if (lldb_private::repro::InstrumentationData _data =                         \
          LLDB_GET_INSTRUMENTATION_DATA()) {                                   \
    _recorder.Record(_data.GetSerializer(), _data.GetRegistry(),               \
                     &lldb_private::repro::invoke<T1>::method<T2>::doit,       \
                     __VA_ARGS__);                                             \
  }

#define LLDB_RECORD_METHOD(Result, Class, Method, Signature, ...)              \
  LLDB_RECORD_(Result(Class::*) Signature, (&Class::Method), this, __VA_ARGS__)

#define LLDB_RECORD_METHOD_CONST(Result, Class, Method, Signature, ...)        \
  LLDB_RECORD_(Result(Class::*) Signature const, (&Class::Method), this,       \
               __VA_ARGS__)

#define LLDB_RECORD_METHOD_NO_ARGS(Result, Class, Method)                      \
  LLDB_RECORD_(Result (Class::*)(), (&Class::Method), this)

#define LLDB_RECORD_METHOD_CONST_NO_ARGS(Result, Class, Method)                \
  LLDB_RECORD_(Result (Class::*)() const, (&Class::Method), this)

#define LLDB_RECORD_STATIC_METHOD(Result, Class, Method, Signature, ...)       \
  LLDB_RECORD_(Result(*) Signature, (&Class::Method), __VA_ARGS__)

#define LLDB_RECORD_STATIC_METHOD_NO_ARGS(Result, Class, Method)               \
  LLDB_RECORD_(Result (*)(), (&Class::Method), lldb_private::repro::EmptyArg())

#define LLDB_RECORD_RESULT(Result) _recorder.RecordResult(Result, true);

/// The LLDB_RECORD_DUMMY macro is special because it doesn't actually record
/// anything. It's used to track API boundaries when we cannot record for
/// technical reasons.
#define LLDB_RECORD_DUMMY(Result, Class, Method, Signature, ...)               \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));
#define LLDB_RECORD_DUMMY_NO_ARGS(Result, Class, Method)                       \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION);

namespace lldb_private {
namespace repro {

template <class T>
struct is_trivially_serializable
    : std::integral_constant<bool, std::is_fundamental<T>::value ||
                                       std::is_enum<T>::value> {};

/// Mapping between serialized indices and their corresponding objects.
///
/// This class is used during replay to map indices back to in-memory objects.
///
/// When objects are constructed, they are added to this mapping using
/// AddObjectForIndex.
///
/// When an object is passed to a function, its index is deserialized and
/// AddObjectForIndex returns the corresponding object. If there is no object
/// for the given index, a nullptr is returend. The latter is valid when custom
/// replay code is in place and the actual object is ignored.
class IndexToObject {
public:
  /// Returns an object as a pointer for the given index or nullptr if not
  /// present in the map.
  template <typename T> T *GetObjectForIndex(unsigned idx) {
    assert(idx != 0 && "Cannot get object for sentinel");
    void *object = GetObjectForIndexImpl(idx);
    return static_cast<T *>(object);
  }

  /// Adds a pointer to an object to the mapping for the given index.
  template <typename T> void AddObjectForIndex(unsigned idx, T *object) {
    AddObjectForIndexImpl(
        idx, static_cast<void *>(
                 const_cast<typename std::remove_const<T>::type *>(object)));
  }

  /// Adds a reference to an object to the mapping for the given index.
  template <typename T> void AddObjectForIndex(unsigned idx, T &object) {
    AddObjectForIndexImpl(
        idx, static_cast<void *>(
                 const_cast<typename std::remove_const<T>::type *>(&object)));
  }

  /// Get all objects sorted by their index.
  std::vector<void *> GetAllObjects() const;

private:
  /// Helper method that does the actual lookup. The void* result is later cast
  /// by the caller.
  void *GetObjectForIndexImpl(unsigned idx);

  /// Helper method that does the actual insertion.
  void AddObjectForIndexImpl(unsigned idx, void *object);

  /// Keeps a mapping between indices and their corresponding object.
  llvm::DenseMap<unsigned, void *> m_mapping;
};

/// We need to differentiate between pointers to fundamental and
/// non-fundamental types. See the corresponding Deserializer::Read method
/// for the reason why.
struct PointerTag {};
struct ReferenceTag {};
struct ValueTag {};
struct FundamentalPointerTag {};
struct FundamentalReferenceTag {};

/// Return the deserialization tag for the given type T.
template <class T> struct serializer_tag {
  typedef typename std::conditional<std::is_trivially_copyable<T>::value,
                                    ValueTag, ReferenceTag>::type type;
};
template <class T> struct serializer_tag<T *> {
  typedef
      typename std::conditional<std::is_fundamental<T>::value,
                                FundamentalPointerTag, PointerTag>::type type;
};
template <class T> struct serializer_tag<T &> {
  typedef typename std::conditional<std::is_fundamental<T>::value,
                                    FundamentalReferenceTag, ReferenceTag>::type
      type;
};

/// Deserializes data from a buffer. It is used to deserialize function indices
/// to replay, their arguments and return values.
///
/// Fundamental types and strings are read by value. Objects are read by their
/// index, which get translated by the IndexToObject mapping maintained in
/// this class.
///
/// Additional bookkeeping with regards to the IndexToObject is required to
/// deserialize objects. When a constructor is run or an object is returned by
/// value, we need to capture the object and add it to the index together with
/// its index. This is the job of HandleReplayResult(Void).
class Deserializer {
public:
  Deserializer(llvm::StringRef buffer) : m_buffer(buffer) {}

  /// Returns true when the buffer has unread data.
  bool HasData(unsigned size) { return size <= m_buffer.size(); }

  /// Deserialize and interpret value as T.
  template <typename T> T Deserialize() {
    T t = Read<T>(typename serializer_tag<T>::type());
#ifdef LLDB_REPRO_INSTR_TRACE
    llvm::errs() << "Deserializing with " << LLVM_PRETTY_FUNCTION << " -> "
                 << stringify_args(t) << "\n";
#endif
    return t;
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> void HandleReplayResult(const T &t) {
    unsigned result = Deserialize<unsigned>();
    if (is_trivially_serializable<T>::value)
      return;
    // We need to make a copy as the original object might go out of scope.
    m_index_to_object.AddObjectForIndex(result, new T(t));
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> void HandleReplayResult(T *t) {
    unsigned result = Deserialize<unsigned>();
    if (is_trivially_serializable<T>::value)
      return;
    m_index_to_object.AddObjectForIndex(result, t);
  }

  /// All returned types are recorded, even when the function returns a void.
  /// The latter requires special handling.
  void HandleReplayResultVoid() {
    unsigned result = Deserialize<unsigned>();
    assert(result == 0);
    (void)result;
  }

  std::vector<void *> GetAllObjects() const {
    return m_index_to_object.GetAllObjects();
  }

private:
  template <typename T> T Read(ValueTag) {
    assert(HasData(sizeof(T)));
    T t;
    std::memcpy(reinterpret_cast<char *>(&t), m_buffer.data(), sizeof(T));
    m_buffer = m_buffer.drop_front(sizeof(T));
    return t;
  }

  template <typename T> T Read(PointerTag) {
    typedef typename std::remove_pointer<T>::type UnderlyingT;
    return m_index_to_object.template GetObjectForIndex<UnderlyingT>(
        Deserialize<unsigned>());
  }

  template <typename T> T Read(ReferenceTag) {
    typedef typename std::remove_reference<T>::type UnderlyingT;
    // If this is a reference to a fundamental type we just read its value.
    return *m_index_to_object.template GetObjectForIndex<UnderlyingT>(
        Deserialize<unsigned>());
  }

  /// This method is used to parse references to fundamental types. Because
  /// they're not recorded in the object table we have serialized their value.
  /// We read its value, allocate a copy on the heap, and return a pointer to
  /// the copy.
  template <typename T> T Read(FundamentalPointerTag) {
    typedef typename std::remove_pointer<T>::type UnderlyingT;
    return new UnderlyingT(Deserialize<UnderlyingT>());
  }

  /// This method is used to parse references to fundamental types. Because
  /// they're not recorded in the object table we have serialized their value.
  /// We read its value, allocate a copy on the heap, and return a reference to
  /// the copy.
  template <typename T> T Read(FundamentalReferenceTag) {
    // If this is a reference to a fundamental type we just read its value.
    typedef typename std::remove_reference<T>::type UnderlyingT;
    return *(new UnderlyingT(Deserialize<UnderlyingT>()));
  }

  /// Mapping of indices to objects.
  IndexToObject m_index_to_object;

  /// Buffer containing the serialized data.
  llvm::StringRef m_buffer;
};

/// Partial specialization for C-style strings. We read the string value
/// instead of treating it as pointer.
template <> const char *Deserializer::Deserialize<const char *>();
template <> const char **Deserializer::Deserialize<const char **>();
template <> const uint8_t *Deserializer::Deserialize<const uint8_t *>();
template <> const void *Deserializer::Deserialize<const void *>();
template <> char *Deserializer::Deserialize<char *>();
template <> void *Deserializer::Deserialize<void *>();

/// Helpers to auto-synthesize function replay code. It deserializes the replay
/// function's arguments one by one and finally calls the corresponding
/// function.
template <typename... Remaining> struct DeserializationHelper;

template <typename Head, typename... Tail>
struct DeserializationHelper<Head, Tail...> {
  template <typename Result, typename... Deserialized> struct deserialized {
    static Result doit(Deserializer &deserializer,
                       Result (*f)(Deserialized..., Head, Tail...),
                       Deserialized... d) {
      return DeserializationHelper<Tail...>::
          template deserialized<Result, Deserialized..., Head>::doit(
              deserializer, f, d..., deserializer.Deserialize<Head>());
    }
  };
};

template <> struct DeserializationHelper<> {
  template <typename Result, typename... Deserialized> struct deserialized {
    static Result doit(Deserializer &deserializer, Result (*f)(Deserialized...),
                       Deserialized... d) {
      return f(d...);
    }
  };
};

/// The replayer interface.
struct Replayer {
  virtual ~Replayer() {}
  virtual void operator()(Deserializer &deserializer) const = 0;
};

/// The default replayer deserializes the arguments and calls the function.
template <typename Signature> struct DefaultReplayer;
template <typename Result, typename... Args>
struct DefaultReplayer<Result(Args...)> : public Replayer {
  DefaultReplayer(Result (*f)(Args...)) : Replayer(), f(f) {}

  void operator()(Deserializer &deserializer) const override {
    deserializer.HandleReplayResult(
        DeserializationHelper<Args...>::template deserialized<Result>::doit(
            deserializer, f));
  }

  Result (*f)(Args...);
};

/// Partial specialization for function returning a void type. It ignores the
/// (absent) return value.
template <typename... Args>
struct DefaultReplayer<void(Args...)> : public Replayer {
  DefaultReplayer(void (*f)(Args...)) : Replayer(), f(f) {}

  void operator()(Deserializer &deserializer) const override {
    DeserializationHelper<Args...>::template deserialized<void>::doit(
        deserializer, f);
    deserializer.HandleReplayResultVoid();
  }

  void (*f)(Args...);
};

/// The registry contains a unique mapping between functions and their ID. The
/// IDs can be serialized and deserialized to replay a function. Functions need
/// to be registered with the registry for this to work.
class Registry {
private:
  struct SignatureStr {
    SignatureStr(llvm::StringRef result = {}, llvm::StringRef scope = {},
                 llvm::StringRef name = {}, llvm::StringRef args = {})
        : result(result), scope(scope), name(name), args(args) {}

    std::string ToString() const;

    llvm::StringRef result;
    llvm::StringRef scope;
    llvm::StringRef name;
    llvm::StringRef args;
  };

public:
  Registry() = default;
  virtual ~Registry() = default;

  /// Register a default replayer for a function.
  template <typename Signature>
  void Register(Signature *f, llvm::StringRef result = {},
                llvm::StringRef scope = {}, llvm::StringRef name = {},
                llvm::StringRef args = {}) {
    DoRegister(uintptr_t(f), std::make_unique<DefaultReplayer<Signature>>(f),
               SignatureStr(result, scope, name, args));
  }

  /// Register a replayer that invokes a custom function with the same
  /// signature as the replayed function.
  template <typename Signature>
  void Register(Signature *f, Signature *g, llvm::StringRef result = {},
                llvm::StringRef scope = {}, llvm::StringRef name = {},
                llvm::StringRef args = {}) {
    DoRegister(uintptr_t(f), std::make_unique<DefaultReplayer<Signature>>(g),
               SignatureStr(result, scope, name, args));
  }

  /// Replay functions from a file.
  bool Replay(const FileSpec &file);

  /// Replay functions from a buffer.
  bool Replay(llvm::StringRef buffer);

  /// Replay functions from a deserializer.
  bool Replay(Deserializer &deserializer);

  /// Returns the ID for a given function address.
  unsigned GetID(uintptr_t addr);

protected:
  /// Register the given replayer for a function (and the ID mapping).
  void DoRegister(uintptr_t RunID, std::unique_ptr<Replayer> replayer,
                  SignatureStr signature);

private:
  std::string GetSignature(unsigned id);
  Replayer *GetReplayer(unsigned id);

  /// Mapping of function addresses to replayers and their ID.
  std::map<uintptr_t, std::pair<std::unique_ptr<Replayer>, unsigned>>
      m_replayers;

  /// Mapping of IDs to replayer instances.
  std::map<unsigned, std::pair<Replayer *, SignatureStr>> m_ids;
};

/// To be used as the "Runtime ID" of a constructor. It also invokes the
/// constructor when called.
template <typename Signature> struct construct;
template <typename Class, typename... Args> struct construct<Class(Args...)> {
  static Class *doit(Args... args) { return new Class(args...); }
};

/// To be used as the "Runtime ID" of a member function. It also invokes the
/// member function when called.
template <typename Signature> struct invoke;
template <typename Result, typename Class, typename... Args>
struct invoke<Result (Class::*)(Args...)> {
  template <Result (Class::*m)(Args...)> struct method {
    static Result doit(Class *c, Args... args) { return (c->*m)(args...); }
  };
};

template <typename Result, typename Class, typename... Args>
struct invoke<Result (Class::*)(Args...) const> {
  template <Result (Class::*m)(Args...) const> struct method {
    static Result doit(Class *c, Args... args) { return (c->*m)(args...); }
  };
};

template <typename Result, typename... Args>
struct invoke<Result (*)(Args...)> {
  template <Result (*m)(Args...)> struct method {
    static Result doit(Args... args) { return (*m)(args...); }
  };
};

template <typename... Args> struct invoke<void (*)(Args...)> {
  template <void (*m)(Args...)> struct method {
    static void doit(Args... args) { return (*m)(args...); }
  };
};

template <typename Class, typename... Args>
struct invoke<void (Class::*)(Args...)> {
  template <void (Class::*m)(Args...)> struct method {
    static void doit(Class *c, Args... args) { (c->*m)(args...); }
  };
};

/// Maps an object to an index for serialization. Indices are unique and
/// incremented for every new object.
///
/// Indices start at 1 in order to differentiate with an invalid index (0) in
/// the serialized buffer.
class ObjectToIndex {
public:
  template <typename T> unsigned GetIndexForObject(T *t) {
    return GetIndexForObjectImpl(static_cast<const void *>(t));
  }

private:
  unsigned GetIndexForObjectImpl(const void *object);

  llvm::DenseMap<const void *, unsigned> m_mapping;
};

/// Serializes functions, their arguments and their return type to a stream.
class Serializer {
public:
  Serializer(llvm::raw_ostream &stream = llvm::outs()) : m_stream(stream) {}

  /// Recursively serialize all the given arguments.
  template <typename Head, typename... Tail>
  void SerializeAll(const Head &head, const Tail &... tail) {
    Serialize(head);
    SerializeAll(tail...);
  }

  void SerializeAll() { m_stream.flush(); }

private:
  /// Serialize pointers. We need to differentiate between pointers to
  /// fundamental types (in which case we serialize its value) and pointer to
  /// objects (in which case we serialize their index).
  template <typename T> void Serialize(T *t) {
#ifdef LLDB_REPRO_INSTR_TRACE
    llvm::errs() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
                 << stringify_args(t) << "\n";
#endif
    if (std::is_fundamental<T>::value) {
      Serialize(*t);
    } else {
      unsigned idx = m_tracker.GetIndexForObject(t);
      Serialize(idx);
    }
  }

  /// Serialize references. We need to differentiate between references to
  /// fundamental types (in which case we serialize its value) and references
  /// to objects (in which case we serialize their index).
  template <typename T> void Serialize(T &t) {
#ifdef LLDB_REPRO_INSTR_TRACE
    llvm::errs() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
                 << stringify_args(t) << "\n";
#endif
    if (is_trivially_serializable<T>::value) {
      m_stream.write(reinterpret_cast<const char *>(&t), sizeof(T));
    } else {
      unsigned idx = m_tracker.GetIndexForObject(&t);
      Serialize(idx);
    }
  }

  void Serialize(const void *v) {
    // FIXME: Support void*
  }

  void Serialize(void *v) {
    // FIXME: Support void*
  }

  void Serialize(const char *t) {
#ifdef LLDB_REPRO_INSTR_TRACE
    llvm::errs() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
                 << stringify_args(t) << "\n";
#endif
    const size_t size = t ? strlen(t) : std::numeric_limits<size_t>::max();
    Serialize(size);
    if (t) {
      m_stream << t;
      m_stream.write(0x0);
    }
  }

  void Serialize(const char **t) {
    size_t size = 0;
    if (!t) {
      Serialize(size);
      return;
    }

    // Compute the size of the array.
    const char *const *temp = t;
    while (*temp++)
      size++;
    Serialize(size);

    // Serialize the content of the array.
    while (*t)
      Serialize(*t++);
  }

  /// Serialization stream.
  llvm::raw_ostream &m_stream;

  /// Mapping of objects to indices.
  ObjectToIndex m_tracker;
};

class InstrumentationData {
public:
  InstrumentationData() : m_serializer(nullptr), m_registry(nullptr){};
  InstrumentationData(Serializer &serializer, Registry &registry)
      : m_serializer(&serializer), m_registry(&registry){};

  Serializer &GetSerializer() { return *m_serializer; }
  Registry &GetRegistry() { return *m_registry; }

  operator bool() { return m_serializer != nullptr && m_registry != nullptr; }

private:
  Serializer *m_serializer;
  Registry *m_registry;
};

struct EmptyArg {};

/// RAII object that records function invocations and their return value.
///
/// API calls are only captured when the API boundary is crossed. Once we're in
/// the API layer, and another API function is called, it doesn't need to be
/// recorded.
///
/// When a call is recored, its result is always recorded as well, even if the
/// function returns a void. For functions that return by value, RecordResult
/// should be used. Otherwise a sentinel value (0) will be serialized.
///
/// Because of the functional overlap between logging and recording API calls,
/// this class is also used for logging.
class Recorder {
public:
  Recorder(llvm::StringRef pretty_func = {}, std::string &&pretty_args = {});
  ~Recorder();

  /// Records a single function call.
  template <typename Result, typename... FArgs, typename... RArgs>
  void Record(Serializer &serializer, Registry &registry, Result (*f)(FArgs...),
              const RArgs &... args) {
    m_serializer = &serializer;
    if (!ShouldCapture())
      return;

    unsigned id = registry.GetID(uintptr_t(f));

#ifdef LLDB_REPRO_INSTR_TRACE
    Log(id);
#endif

    serializer.SerializeAll(id);
    serializer.SerializeAll(args...);

    if (std::is_class<typename std::remove_pointer<
            typename std::remove_reference<Result>::type>::type>::value) {
      m_result_recorded = false;
    } else {
      serializer.SerializeAll(0);
      m_result_recorded = true;
    }
  }

  /// Records a single function call.
  template <typename... Args>
  void Record(Serializer &serializer, Registry &registry, void (*f)(Args...),
              const Args &... args) {
    m_serializer = &serializer;
    if (!ShouldCapture())
      return;

    unsigned id = registry.GetID(uintptr_t(f));

#ifdef LLDB_REPRO_INSTR_TRACE
    Log(id);
#endif

    serializer.SerializeAll(id);
    serializer.SerializeAll(args...);

    // Record result.
    serializer.SerializeAll(0);
    m_result_recorded = true;
  }

  /// Specializations for the no-argument methods. These are passed an empty
  /// dummy argument so the same variadic macro can be used. These methods
  /// strip the arguments before forwarding them.
  template <typename Result>
  void Record(Serializer &serializer, Registry &registry, Result (*f)(),
              const EmptyArg &arg) {
    Record(serializer, registry, f);
  }

  /// Record the result of a function call.
  template <typename Result>
  Result RecordResult(Result &&r, bool update_boundary) {
    // When recording the result from the LLDB_RECORD_RESULT macro, we need to
    // update the boundary so we capture the copy constructor. However, when
    // called to record the this pointer of the (copy) constructor, the
    // boundary should not be toggled, because it is called from the
    // LLDB_RECORD_CONSTRUCTOR macro, which might be followed by other API
    // calls.
    if (update_boundary)
      UpdateBoundary();
    if (m_serializer && ShouldCapture()) {
      assert(!m_result_recorded);
      m_serializer->SerializeAll(r);
      m_result_recorded = true;
    }
    return std::forward<Result>(r);
  }

private:
  void UpdateBoundary() {
    if (m_local_boundary)
      g_global_boundary = false;
  }

  bool ShouldCapture() { return m_local_boundary; }

#ifdef LLDB_REPRO_INSTR_TRACE
  void Log(unsigned id) {
    llvm::errs() << "Recording " << id << ": " << m_pretty_func << " ("
                 << m_pretty_args << ")\n";
  }
#endif

  Serializer *m_serializer;

  /// Pretty function for logging.
  llvm::StringRef m_pretty_func;
  std::string m_pretty_args;

  /// Whether this function call was the one crossing the API boundary.
  bool m_local_boundary;

  /// Whether the return value was recorded explicitly.
  bool m_result_recorded;

  /// Whether we're currently across the API boundary.
  static bool g_global_boundary;
};

template <typename Signature> struct char_ptr_redirect;
template <typename Result, typename Class>
struct char_ptr_redirect<Result (Class::*)(char *, size_t) const> {
  template <Result (Class::*m)(char *, size_t) const> struct method {
    static Result doit(Class *c, char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (c->*m)(buffer, l);
    }
  };
};
template <typename Result, typename Class>
struct char_ptr_redirect<Result (Class::*)(char *, size_t)> {
  template <Result (Class::*m)(char *, size_t)> struct method {
    static Result doit(Class *c, char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (c->*m)(buffer, l);
    }
  };
};

template <typename Result>
struct char_ptr_redirect<Result (*)(char *, size_t)> {
  template <Result (*m)(char *, size_t)> struct method {
    static Result doit(char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (*m)(buffer, l);
    }
  };
};

} // namespace repro
} // namespace lldb_private

#endif // LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H
