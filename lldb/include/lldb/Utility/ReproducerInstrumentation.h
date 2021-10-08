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

#include <map>
#include <thread>
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

#ifdef LLDB_REPRO_INSTR_TRACE
inline llvm::raw_ostream &this_thread_id() {
  size_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return llvm::errs().write_hex(tid) << " :: ";
}
#endif

#define LLDB_REGISTER_CONSTRUCTOR(Class, Signature)                            \
  R.Register<Class * Signature>(&construct<Class Signature>::record, "",       \
                                #Class, #Class, #Signature)

#define LLDB_REGISTER_METHOD(Result, Class, Method, Signature)                 \
  R.Register(                                                                  \
      &invoke<Result(Class::*) Signature>::method<(&Class::Method)>::record,   \
      #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_METHOD_CONST(Result, Class, Method, Signature)           \
  R.Register(&invoke<Result(Class::*)                                          \
                         Signature const>::method<(&Class::Method)>::record,   \
             #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_STATIC_METHOD(Result, Class, Method, Signature)          \
  R.Register(&invoke<Result(*) Signature>::method<(&Class::Method)>::record,   \
             #Result, #Class, #Method, #Signature)

#define LLDB_REGISTER_CHAR_PTR_METHOD_STATIC(Result, Class, Method)            \
  R.Register(                                                                  \
      &invoke<Result (*)(char *, size_t)>::method<(&Class::Method)>::record,   \
      &invoke_char_ptr<Result (*)(char *,                                      \
                                  size_t)>::method<(&Class::Method)>::record,  \
      #Result, #Class, #Method, "(char*, size_t");

#define LLDB_REGISTER_CHAR_PTR_METHOD(Result, Class, Method)                   \
  R.Register(&invoke<Result (Class::*)(char *, size_t)>::method<(              \
                 &Class::Method)>::record,                                     \
             &invoke_char_ptr<Result (Class::*)(char *, size_t)>::method<(     \
                 &Class::Method)>::record,                                     \
             #Result, #Class, #Method, "(char*, size_t");

#define LLDB_REGISTER_CHAR_PTR_METHOD_CONST(Result, Class, Method)             \
  R.Register(&invoke<Result (Class::*)(char *, size_t)                         \
                         const>::method<(&Class::Method)>::record,             \
             &invoke_char_ptr<Result (Class::*)(char *, size_t)                \
                                  const>::method<(&Class::Method)>::record,    \
             #Result, #Class, #Method, "(char*, size_t");

#define LLDB_CONSTRUCT_(T, Class, ...)                                         \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION);               \
  lldb_private::repro::construct<T>::handle(LLDB_GET_INSTRUMENTATION_DATA(),   \
                                            _recorder, Class, __VA_ARGS__);

#define LLDB_RECORD_CONSTRUCTOR(Class, Signature, ...)                         \
  LLDB_CONSTRUCT_(Class Signature, this, __VA_ARGS__)

#define LLDB_RECORD_CONSTRUCTOR_NO_ARGS(Class)                                 \
  LLDB_CONSTRUCT_(Class(), this, lldb_private::repro::EmptyArg())

#define LLDB_RECORD_(T1, T2, ...)                                              \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));        \
  if (lldb_private::repro::InstrumentationData _data =                         \
          LLDB_GET_INSTRUMENTATION_DATA()) {                                   \
    if (lldb_private::repro::Serializer *_serializer =                         \
            _data.GetSerializer()) {                                           \
      _recorder.Record(*_serializer, _data.GetRegistry(),                      \
                       &lldb_private::repro::invoke<T1>::method<T2>::record,   \
                       __VA_ARGS__);                                           \
    } else if (lldb_private::repro::Deserializer *_deserializer =              \
                   _data.GetDeserializer()) {                                  \
      if (_recorder.ShouldCapture()) {                                         \
        return lldb_private::repro::invoke<T1>::method<T2>::replay(            \
            _recorder, *_deserializer, _data.GetRegistry());                   \
      }                                                                        \
    }                                                                          \
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

#define LLDB_RECORD_CHAR_PTR_(T1, T2, StrOut, ...)                             \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));        \
  if (lldb_private::repro::InstrumentationData _data =                         \
          LLDB_GET_INSTRUMENTATION_DATA()) {                                   \
    if (lldb_private::repro::Serializer *_serializer =                         \
            _data.GetSerializer()) {                                           \
      _recorder.Record(*_serializer, _data.GetRegistry(),                      \
                       &lldb_private::repro::invoke<T1>::method<(T2)>::record, \
                       __VA_ARGS__);                                           \
    } else if (lldb_private::repro::Deserializer *_deserializer =              \
                   _data.GetDeserializer()) {                                  \
      if (_recorder.ShouldCapture()) {                                         \
        return lldb_private::repro::invoke_char_ptr<T1>::method<T2>::replay(   \
            _recorder, *_deserializer, _data.GetRegistry(), StrOut);           \
      }                                                                        \
    }                                                                          \
  }

#define LLDB_RECORD_CHAR_PTR_METHOD(Result, Class, Method, Signature, StrOut,  \
                                    ...)                                       \
  LLDB_RECORD_CHAR_PTR_(Result(Class::*) Signature, (&Class::Method), StrOut,  \
                        this, __VA_ARGS__)

#define LLDB_RECORD_CHAR_PTR_METHOD_CONST(Result, Class, Method, Signature,    \
                                          StrOut, ...)                         \
  LLDB_RECORD_CHAR_PTR_(Result(Class::*) Signature const, (&Class::Method),    \
                        StrOut, this, __VA_ARGS__)

#define LLDB_RECORD_CHAR_PTR_STATIC_METHOD(Result, Class, Method, Signature,   \
                                           StrOut, ...)                        \
  LLDB_RECORD_CHAR_PTR_(Result(*) Signature, (&Class::Method), StrOut,         \
                        __VA_ARGS__)

#define LLDB_RECORD_RESULT(Result) _recorder.RecordResult(Result, true);

/// The LLDB_RECORD_DUMMY macro is special because it doesn't actually record
/// anything. It's used to track API boundaries when we cannot record for
/// technical reasons.
#define LLDB_RECORD_DUMMY(Result, Class, Method, Signature, ...)               \
  lldb_private::repro::Recorder _recorder;

#define LLDB_RECORD_DUMMY_NO_ARGS(Result, Class, Method)                       \
  lldb_private::repro::Recorder _recorder;

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
  template <typename T> T *AddObjectForIndex(unsigned idx, T *object) {
    AddObjectForIndexImpl(
        idx, static_cast<void *>(
                 const_cast<typename std::remove_const<T>::type *>(object)));
    return object;
  }

  /// Adds a reference to an object to the mapping for the given index.
  template <typename T> T &AddObjectForIndex(unsigned idx, T &object) {
    AddObjectForIndexImpl(
        idx, static_cast<void *>(
                 const_cast<typename std::remove_const<T>::type *>(&object)));
    return object;
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

  template <typename T> const T &HandleReplayResult(const T &t) {
    CheckSequence(Deserialize<unsigned>());
    unsigned result = Deserialize<unsigned>();
    if (is_trivially_serializable<T>::value)
      return t;
    // We need to make a copy as the original object might go out of scope.
    return *m_index_to_object.AddObjectForIndex(result, new T(t));
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> T &HandleReplayResult(T &t) {
    CheckSequence(Deserialize<unsigned>());
    unsigned result = Deserialize<unsigned>();
    if (is_trivially_serializable<T>::value)
      return t;
    // We need to make a copy as the original object might go out of scope.
    return *m_index_to_object.AddObjectForIndex(result, new T(t));
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> T *HandleReplayResult(T *t) {
    CheckSequence(Deserialize<unsigned>());
    unsigned result = Deserialize<unsigned>();
    if (is_trivially_serializable<T>::value)
      return t;
    return m_index_to_object.AddObjectForIndex(result, t);
  }

  /// All returned types are recorded, even when the function returns a void.
  /// The latter requires special handling.
  void HandleReplayResultVoid() {
    CheckSequence(Deserialize<unsigned>());
    unsigned result = Deserialize<unsigned>();
    assert(result == 0);
    (void)result;
  }

  std::vector<void *> GetAllObjects() const {
    return m_index_to_object.GetAllObjects();
  }

  void SetExpectedSequence(unsigned sequence) {
    m_expected_sequence = sequence;
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

  /// Verify that the given sequence number matches what we expect.
  void CheckSequence(unsigned sequence);

  /// Mapping of indices to objects.
  IndexToObject m_index_to_object;

  /// Buffer containing the serialized data.
  llvm::StringRef m_buffer;

  /// The result's expected sequence number.
  llvm::Optional<unsigned> m_expected_sequence;
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
  virtual ~Replayer() = default;
  virtual void operator()(Deserializer &deserializer) const = 0;
};

/// The default replayer deserializes the arguments and calls the function.
template <typename Signature> struct DefaultReplayer;
template <typename Result, typename... Args>
struct DefaultReplayer<Result(Args...)> : public Replayer {
  DefaultReplayer(Result (*f)(Args...)) : Replayer(), f(f) {}

  void operator()(Deserializer &deserializer) const override {
    Replay(deserializer);
  }

  Result Replay(Deserializer &deserializer) const {
    return deserializer.HandleReplayResult(
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
    Replay(deserializer);
  }

  void Replay(Deserializer &deserializer) const {
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

  /// Get the replayer matching the given ID.
  Replayer *GetReplayer(unsigned id);

  std::string GetSignature(unsigned id);

  void CheckID(unsigned expected, unsigned actual);

protected:
  /// Register the given replayer for a function (and the ID mapping).
  void DoRegister(uintptr_t RunID, std::unique_ptr<Replayer> replayer,
                  SignatureStr signature);

private:
  /// Mapping of function addresses to replayers and their ID.
  std::map<uintptr_t, std::pair<std::unique_ptr<Replayer>, unsigned>>
      m_replayers;

  /// Mapping of IDs to replayer instances.
  std::map<unsigned, std::pair<Replayer *, SignatureStr>> m_ids;
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
    this_thread_id() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
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
    this_thread_id() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
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
    this_thread_id() << "Serializing with " << LLVM_PRETTY_FUNCTION << " -> "
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
}; // namespace repro

class InstrumentationData {
public:
  Serializer *GetSerializer() { return m_serializer; }
  Deserializer *GetDeserializer() { return m_deserializer; }
  Registry &GetRegistry() { return *m_registry; }

  operator bool() {
    return (m_serializer != nullptr || m_deserializer != nullptr) &&
           m_registry != nullptr;
  }

  static void Initialize(Serializer &serializer, Registry &registry);
  static void Initialize(Deserializer &serializer, Registry &registry);
  static InstrumentationData &Instance();

protected:
  friend llvm::optional_detail::OptionalStorage<InstrumentationData, true>;
  friend llvm::Optional<InstrumentationData>;

  InstrumentationData() = default;
  InstrumentationData(Serializer &serializer, Registry &registry)
      : m_serializer(&serializer), m_deserializer(nullptr),
        m_registry(&registry) {}
  InstrumentationData(Deserializer &deserializer, Registry &registry)
      : m_serializer(nullptr), m_deserializer(&deserializer),
        m_registry(&registry) {}

private:
  static llvm::Optional<InstrumentationData> &InstanceImpl();

  Serializer *m_serializer = nullptr;
  Deserializer *m_deserializer = nullptr;
  Registry *m_registry = nullptr;
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
  Recorder();
  Recorder(llvm::StringRef pretty_func, std::string &&pretty_args = {});
  ~Recorder();

  /// Records a single function call.
  template <typename Result, typename... FArgs, typename... RArgs>
  void Record(Serializer &serializer, Registry &registry, Result (*f)(FArgs...),
              const RArgs &... args) {
    m_serializer = &serializer;
    if (!ShouldCapture())
      return;

    std::lock_guard<std::mutex> lock(g_mutex);
    unsigned sequence = GetSequenceNumber();
    unsigned id = registry.GetID(uintptr_t(f));

#ifdef LLDB_REPRO_INSTR_TRACE
    Log(id);
#endif

    serializer.SerializeAll(sequence);
    serializer.SerializeAll(id);
    serializer.SerializeAll(args...);

    if (std::is_class<typename std::remove_pointer<
            typename std::remove_reference<Result>::type>::type>::value) {
      m_result_recorded = false;
    } else {
      serializer.SerializeAll(sequence);
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

    std::lock_guard<std::mutex> lock(g_mutex);
    unsigned sequence = GetSequenceNumber();
    unsigned id = registry.GetID(uintptr_t(f));

#ifdef LLDB_REPRO_INSTR_TRACE
    Log(id);
#endif

    serializer.SerializeAll(sequence);
    serializer.SerializeAll(id);
    serializer.SerializeAll(args...);

    // Record result.
    serializer.SerializeAll(sequence);
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
      std::lock_guard<std::mutex> lock(g_mutex);
      assert(!m_result_recorded);
      m_serializer->SerializeAll(GetSequenceNumber());
      m_serializer->SerializeAll(r);
      m_result_recorded = true;
    }
    return std::forward<Result>(r);
  }

  template <typename Result, typename T>
  Result Replay(Deserializer &deserializer, Registry &registry, uintptr_t addr,
                bool update_boundary) {
    deserializer.SetExpectedSequence(deserializer.Deserialize<unsigned>());
    unsigned actual_id = registry.GetID(addr);
    unsigned id = deserializer.Deserialize<unsigned>();
    registry.CheckID(id, actual_id);
    return ReplayResult<Result>(
        static_cast<DefaultReplayer<T> *>(registry.GetReplayer(id))
            ->Replay(deserializer),
        update_boundary);
  }

  void Replay(Deserializer &deserializer, Registry &registry, uintptr_t addr) {
    deserializer.SetExpectedSequence(deserializer.Deserialize<unsigned>());
    unsigned actual_id = registry.GetID(addr);
    unsigned id = deserializer.Deserialize<unsigned>();
    registry.CheckID(id, actual_id);
    registry.GetReplayer(id)->operator()(deserializer);
  }

  template <typename Result>
  Result ReplayResult(Result &&r, bool update_boundary) {
    if (update_boundary)
      UpdateBoundary();
    return std::forward<Result>(r);
  }

  bool ShouldCapture() { return m_local_boundary; }

  /// Mark the current thread as a private thread and pretend that everything
  /// on this thread is behind happening behind the API boundary.
  static void PrivateThread();

private:
  static unsigned GetNextSequenceNumber() { return g_sequence++; }
  unsigned GetSequenceNumber() const;

  template <typename T> friend struct replay;
  void UpdateBoundary();

#ifdef LLDB_REPRO_INSTR_TRACE
  void Log(unsigned id) {
    this_thread_id() << "Recording " << id << ": " << m_pretty_func << " ("
                     << m_pretty_args << ")\n";
  }
#endif

  Serializer *m_serializer = nullptr;

  /// Pretty function for logging.
  llvm::StringRef m_pretty_func;
  std::string m_pretty_args;

  /// Whether this function call was the one crossing the API boundary.
  bool m_local_boundary = false;

  /// Whether the return value was recorded explicitly.
  bool m_result_recorded = true;

  /// The sequence number for this pair of function and result.
  unsigned m_sequence;

  /// Global mutex to protect concurrent access.
  static std::mutex g_mutex;

  /// Unique, monotonically increasing sequence number.
  static std::atomic<unsigned> g_sequence;
};

/// To be used as the "Runtime ID" of a constructor. It also invokes the
/// constructor when called.
template <typename Signature> struct construct;
template <typename Class, typename... Args> struct construct<Class(Args...)> {
  static Class *handle(lldb_private::repro::InstrumentationData data,
                       lldb_private::repro::Recorder &recorder, Class *c,
                       const EmptyArg &) {
    return handle(data, recorder, c);
  }

  static Class *handle(lldb_private::repro::InstrumentationData data,
                       lldb_private::repro::Recorder &recorder, Class *c,
                       Args... args) {
    if (!data)
      return nullptr;

    if (Serializer *serializer = data.GetSerializer()) {
      recorder.Record(*serializer, data.GetRegistry(), &record, args...);
      recorder.RecordResult(c, false);
    } else if (Deserializer *deserializer = data.GetDeserializer()) {
      if (recorder.ShouldCapture()) {
        replay(recorder, *deserializer, data.GetRegistry());
      }
    }

    return nullptr;
  }

  static Class *record(Args... args) { return new Class(args...); }

  static Class *replay(Recorder &recorder, Deserializer &deserializer,
                       Registry &registry) {
    return recorder.Replay<Class *, Class *(Args...)>(
        deserializer, registry, uintptr_t(&record), false);
  }
};

/// To be used as the "Runtime ID" of a member function. It also invokes the
/// member function when called.
template <typename Signature> struct invoke;
template <typename Result, typename Class, typename... Args>
struct invoke<Result (Class::*)(Args...)> {
  template <Result (Class::*m)(Args...)> struct method {
    static Result record(Class *c, Args... args) { return (c->*m)(args...); }

    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry) {
      return recorder.Replay<Result, Result(Class *, Args...)>(
          deserializer, registry, uintptr_t(&record), true);
    }
  };
};

template <typename Class, typename... Args>
struct invoke<void (Class::*)(Args...)> {
  template <void (Class::*m)(Args...)> struct method {
    static void record(Class *c, Args... args) { (c->*m)(args...); }
    static void replay(Recorder &recorder, Deserializer &deserializer,
                       Registry &registry) {
      recorder.Replay(deserializer, registry, uintptr_t(&record));
    }
  };
};

template <typename Result, typename Class, typename... Args>
struct invoke<Result (Class::*)(Args...) const> {
  template <Result (Class::*m)(Args...) const> struct method {
    static Result record(Class *c, Args... args) { return (c->*m)(args...); }
    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry) {
      return recorder.Replay<Result, Result(Class *, Args...)>(
          deserializer, registry, uintptr_t(&record), true);
    }
  };
};

template <typename Class, typename... Args>
struct invoke<void (Class::*)(Args...) const> {
  template <void (Class::*m)(Args...) const> struct method {
    static void record(Class *c, Args... args) { return (c->*m)(args...); }
    static void replay(Recorder &recorder, Deserializer &deserializer,
                       Registry &registry) {
      recorder.Replay(deserializer, registry, uintptr_t(&record));
    }
  };
};

template <typename Signature> struct replay;

template <typename Result, typename Class, typename... Args>
struct replay<Result (Class::*)(Args...)> {
  template <Result (Class::*m)(Args...)> struct method {};
};

template <typename Result, typename... Args>
struct invoke<Result (*)(Args...)> {
  template <Result (*m)(Args...)> struct method {
    static Result record(Args... args) { return (*m)(args...); }
    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry) {
      return recorder.Replay<Result, Result(Args...)>(deserializer, registry,
                                                      uintptr_t(&record), true);
    }
  };
};

template <typename... Args> struct invoke<void (*)(Args...)> {
  template <void (*m)(Args...)> struct method {
    static void record(Args... args) { return (*m)(args...); }
    static void replay(Recorder &recorder, Deserializer &deserializer,
                       Registry &registry) {
      recorder.Replay(deserializer, registry, uintptr_t(&record));
    }
  };
};

/// Special handling for functions returning strings as (char*, size_t).
/// {

/// For inline replay, we ignore the arguments and use the ones from the
/// serializer instead. This doesn't work for methods that use a char* and a
/// size to return a string. For one these functions have a custom replayer to
/// prevent override the input buffer. Furthermore, the template-generated
/// deserialization is not easy to hook into.
///
/// The specializations below hand-implement the serialization logic for the
/// inline replay. Instead of using the function from the registry, it uses the
/// one passed into the macro.
template <typename Signature> struct invoke_char_ptr;
template <typename Result, typename Class, typename... Args>
struct invoke_char_ptr<Result (Class::*)(Args...) const> {
  template <Result (Class::*m)(Args...) const> struct method {
    static Result record(Class *c, char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (c->*m)(buffer, l);
    }

    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry, char *str) {
      deserializer.SetExpectedSequence(deserializer.Deserialize<unsigned>());
      deserializer.Deserialize<unsigned>();
      Class *c = deserializer.Deserialize<Class *>();
      deserializer.Deserialize<const char *>();
      size_t l = deserializer.Deserialize<size_t>();
      return recorder.ReplayResult(
          std::move(deserializer.HandleReplayResult((c->*m)(str, l))), true);
    }
  };
};

template <typename Signature> struct invoke_char_ptr;
template <typename Result, typename Class, typename... Args>
struct invoke_char_ptr<Result (Class::*)(Args...)> {
  template <Result (Class::*m)(Args...)> struct method {
    static Result record(Class *c, char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (c->*m)(buffer, l);
    }

    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry, char *str) {
      deserializer.SetExpectedSequence(deserializer.Deserialize<unsigned>());
      deserializer.Deserialize<unsigned>();
      Class *c = deserializer.Deserialize<Class *>();
      deserializer.Deserialize<const char *>();
      size_t l = deserializer.Deserialize<size_t>();
      return recorder.ReplayResult(
          std::move(deserializer.HandleReplayResult((c->*m)(str, l))), true);
    }
  };
};

template <typename Result, typename... Args>
struct invoke_char_ptr<Result (*)(Args...)> {
  template <Result (*m)(Args...)> struct method {
    static Result record(char *s, size_t l) {
      char *buffer = reinterpret_cast<char *>(calloc(l, sizeof(char)));
      return (*m)(buffer, l);
    }

    static Result replay(Recorder &recorder, Deserializer &deserializer,
                         Registry &registry, char *str) {
      deserializer.SetExpectedSequence(deserializer.Deserialize<unsigned>());
      deserializer.Deserialize<unsigned>();
      deserializer.Deserialize<const char *>();
      size_t l = deserializer.Deserialize<size_t>();
      return recorder.ReplayResult(
          std::move(deserializer.HandleReplayResult((*m)(str, l))), true);
    }
  };
};
/// }

} // namespace repro
} // namespace lldb_private

#endif // LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H
