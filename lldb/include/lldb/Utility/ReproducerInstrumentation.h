//===-- ReproducerInstrumentation.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REPRODUCER_INSTRUMENTATION_H
#define LLDB_UTILITY_REPRODUCER_INSTRUMENTATION_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <map>
#include <mutex>

namespace lldb_private {
namespace repro {

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
template <class T> struct serializer_tag { typedef ValueTag type; };
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
    return Read<T>(typename serializer_tag<T>::type());
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> void HandleReplayResult(const T &t) {
    unsigned result = Deserialize<unsigned>();
    if (std::is_fundamental<T>::value)
      return;
    // We need to make a copy as the original object might go out of scope.
    m_index_to_object.AddObjectForIndex(result, new T(t));
  }

  /// Store the returned value in the index-to-object mapping.
  template <typename T> void HandleReplayResult(T *t) {
    unsigned result = Deserialize<unsigned>();
    if (std::is_fundamental<T>::value)
      return;
    m_index_to_object.AddObjectForIndex(result, t);
  }

  /// All returned types are recorded, even when the function returns a void.
  /// The latter requires special handling.
  void HandleReplayResultVoid() {
    unsigned result = Deserialize<unsigned>();
    assert(result == 0);
  }

protected:
  IndexToObject &GetIndexToObject() { return m_index_to_object; }

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
template <> char *Deserializer::Deserialize<char *>();

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

/// Maps an object to an index for serialization. Indices are unique and
/// incremented for every new object.
///
/// Indices start at 1 in order to differentiate with an invalid index (0) in
/// the serialized buffer.
class ObjectToIndex {
public:
  template <typename T> unsigned GetIndexForObject(T *t) {
    return GetIndexForObjectImpl((void *)t);
  }

private:
  unsigned GetIndexForObjectImpl(void *object);

  std::mutex m_mutex;
  llvm::DenseMap<void *, unsigned> m_mapping;
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

  void SerializeAll() {}

private:
  /// Serialize pointers. We need to differentiate between pointers to
  /// fundamental types (in which case we serialize its value) and pointer to
  /// objects (in which case we serialize their index).
  template <typename T> void Serialize(T *t) {
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
    if (std::is_fundamental<T>::value) {
      m_stream.write(reinterpret_cast<const char *>(&t), sizeof(T));
    } else {
      unsigned idx = m_tracker.GetIndexForObject(&t);
      Serialize(idx);
    }
  }

  void Serialize(void *v) {
    // FIXME: Support void*
    llvm_unreachable("void* is currently unsupported.");
  }

  void Serialize(const char *t) {
    m_stream << t;
    m_stream.write(0x0);
  }

  /// Serialization stream.
  llvm::raw_ostream &m_stream;

  /// Mapping of objects to indices.
  ObjectToIndex m_tracker;
};

} // namespace repro
} // namespace lldb_private

#endif // LLDB_UTILITY_REPRODUCER_INSTRUMENTATION_H
