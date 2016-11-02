//===-- StructuredData.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StructuredData_h_
#define liblldb_StructuredData_h_

// C Includes
// C++ Includes
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/StringRef.h"

// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/lldb-defines.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class StructuredData StructuredData.h "lldb/Core/StructuredData.h"
/// @brief A class which can hold structured data
///
/// The StructuredData class is designed to hold the data from a JSON
/// or plist style file -- a serialized data structure with dictionaries
/// (maps, hashes), arrays, and concrete values like integers, floating
/// point numbers, strings, booleans.
///
/// StructuredData does not presuppose any knowledge of the schema for
/// the data it is holding; it can parse JSON data, for instance, and
/// other parts of lldb can iterate through the parsed data set to find
/// keys and values that may be present.
//----------------------------------------------------------------------

class StructuredData {
public:
  class Object;
  class Array;
  class Integer;
  class Float;
  class Boolean;
  class String;
  class Dictionary;
  class Generic;

  typedef std::shared_ptr<Object> ObjectSP;
  typedef std::shared_ptr<Array> ArraySP;
  typedef std::shared_ptr<Integer> IntegerSP;
  typedef std::shared_ptr<Float> FloatSP;
  typedef std::shared_ptr<Boolean> BooleanSP;
  typedef std::shared_ptr<String> StringSP;
  typedef std::shared_ptr<Dictionary> DictionarySP;
  typedef std::shared_ptr<Generic> GenericSP;

  enum class Type {
    eTypeInvalid = -1,
    eTypeNull = 0,
    eTypeGeneric,
    eTypeArray,
    eTypeInteger,
    eTypeFloat,
    eTypeBoolean,
    eTypeString,
    eTypeDictionary
  };

  class Object : public std::enable_shared_from_this<Object> {
  public:
    Object(Type t = Type::eTypeInvalid) : m_type(t) {}

    virtual ~Object() = default;

    virtual bool IsValid() const { return true; }

    virtual void Clear() { m_type = Type::eTypeInvalid; }

    Type GetType() const { return m_type; }

    void SetType(Type t) { m_type = t; }

    Array *GetAsArray() {
      return ((m_type == Type::eTypeArray) ? static_cast<Array *>(this)
                                           : nullptr);
    }

    Dictionary *GetAsDictionary() {
      return ((m_type == Type::eTypeDictionary)
                  ? static_cast<Dictionary *>(this)
                  : nullptr);
    }

    Integer *GetAsInteger() {
      return ((m_type == Type::eTypeInteger) ? static_cast<Integer *>(this)
                                             : nullptr);
    }

    uint64_t GetIntegerValue(uint64_t fail_value = 0) {
      Integer *integer = GetAsInteger();
      return ((integer != nullptr) ? integer->GetValue() : fail_value);
    }

    Float *GetAsFloat() {
      return ((m_type == Type::eTypeFloat) ? static_cast<Float *>(this)
                                           : nullptr);
    }

    double GetFloatValue(double fail_value = 0.0) {
      Float *f = GetAsFloat();
      return ((f != nullptr) ? f->GetValue() : fail_value);
    }

    Boolean *GetAsBoolean() {
      return ((m_type == Type::eTypeBoolean) ? static_cast<Boolean *>(this)
                                             : nullptr);
    }

    bool GetBooleanValue(bool fail_value = false) {
      Boolean *b = GetAsBoolean();
      return ((b != nullptr) ? b->GetValue() : fail_value);
    }

    String *GetAsString() {
      return ((m_type == Type::eTypeString) ? static_cast<String *>(this)
                                            : nullptr);
    }

    std::string GetStringValue(const char *fail_value = nullptr) {
      String *s = GetAsString();
      if (s)
        return s->GetValue();

      if (fail_value && fail_value[0])
        return std::string(fail_value);

      return std::string();
    }

    Generic *GetAsGeneric() {
      return ((m_type == Type::eTypeGeneric) ? static_cast<Generic *>(this)
                                             : nullptr);
    }

    ObjectSP GetObjectForDotSeparatedPath(llvm::StringRef path);

    void DumpToStdout(bool pretty_print = true) const;

    virtual void Dump(Stream &s, bool pretty_print = true) const = 0;

  private:
    Type m_type;
  };

  class Array : public Object {
  public:
    Array() : Object(Type::eTypeArray) {}

    ~Array() override = default;

    bool
    ForEach(std::function<bool(Object *object)> const &foreach_callback) const {
      for (const auto &object_sp : m_items) {
        if (foreach_callback(object_sp.get()) == false)
          return false;
      }
      return true;
    }

    size_t GetSize() const { return m_items.size(); }

    ObjectSP operator[](size_t idx) {
      if (idx < m_items.size())
        return m_items[idx];
      return ObjectSP();
    }

    ObjectSP GetItemAtIndex(size_t idx) const {
      assert(idx < GetSize());
      if (idx < m_items.size())
        return m_items[idx];
      return ObjectSP();
    }

    template <class IntType>
    bool GetItemAtIndexAsInteger(size_t idx, IntType &result) const {
      ObjectSP value_sp = GetItemAtIndex(idx);
      if (value_sp.get()) {
        if (auto int_value = value_sp->GetAsInteger()) {
          result = static_cast<IntType>(int_value->GetValue());
          return true;
        }
      }
      return false;
    }

    template <class IntType>
    bool GetItemAtIndexAsInteger(size_t idx, IntType &result,
                                 IntType default_val) const {
      bool success = GetItemAtIndexAsInteger(idx, result);
      if (!success)
        result = default_val;
      return success;
    }

    bool GetItemAtIndexAsString(size_t idx, std::string &result) const {
      ObjectSP value_sp = GetItemAtIndex(idx);
      if (value_sp.get()) {
        if (auto string_value = value_sp->GetAsString()) {
          result = string_value->GetValue();
          return true;
        }
      }
      return false;
    }

    bool GetItemAtIndexAsString(size_t idx, std::string &result,
                                const std::string &default_val) const {
      bool success = GetItemAtIndexAsString(idx, result);
      if (!success)
        result = default_val;
      return success;
    }

    bool GetItemAtIndexAsString(size_t idx, ConstString &result) const {
      ObjectSP value_sp = GetItemAtIndex(idx);
      if (value_sp.get()) {
        if (auto string_value = value_sp->GetAsString()) {
          result = ConstString(string_value->GetValue());
          return true;
        }
      }
      return false;
    }

    bool GetItemAtIndexAsString(size_t idx, ConstString &result,
                                const char *default_val) const {
      bool success = GetItemAtIndexAsString(idx, result);
      if (!success)
        result.SetCString(default_val);
      return success;
    }

    bool GetItemAtIndexAsDictionary(size_t idx, Dictionary *&result) const {
      result = nullptr;
      ObjectSP value_sp = GetItemAtIndex(idx);
      if (value_sp.get()) {
        result = value_sp->GetAsDictionary();
        return (result != nullptr);
      }
      return false;
    }

    bool GetItemAtIndexAsArray(size_t idx, Array *&result) const {
      result = nullptr;
      ObjectSP value_sp = GetItemAtIndex(idx);
      if (value_sp.get()) {
        result = value_sp->GetAsArray();
        return (result != nullptr);
      }
      return false;
    }

    void Push(ObjectSP item) { m_items.push_back(item); }

    void AddItem(ObjectSP item) { m_items.push_back(item); }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    typedef std::vector<ObjectSP> collection;
    collection m_items;
  };

  class Integer : public Object {
  public:
    Integer(uint64_t i = 0) : Object(Type::eTypeInteger), m_value(i) {}

    ~Integer() override = default;

    void SetValue(uint64_t value) { m_value = value; }

    uint64_t GetValue() { return m_value; }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    uint64_t m_value;
  };

  class Float : public Object {
  public:
    Float(double d = 0.0) : Object(Type::eTypeFloat), m_value(d) {}

    ~Float() override = default;

    void SetValue(double value) { m_value = value; }

    double GetValue() { return m_value; }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    double m_value;
  };

  class Boolean : public Object {
  public:
    Boolean(bool b = false) : Object(Type::eTypeBoolean), m_value(b) {}

    ~Boolean() override = default;

    void SetValue(bool value) { m_value = value; }

    bool GetValue() { return m_value; }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    bool m_value;
  };

  class String : public Object {
  public:
    String(const char *cstr = nullptr) : Object(Type::eTypeString), m_value() {
      if (cstr)
        m_value = cstr;
    }

    String(const std::string &s) : Object(Type::eTypeString), m_value(s) {}

    String(const std::string &&s) : Object(Type::eTypeString), m_value(s) {}

    void SetValue(const std::string &string) { m_value = string; }

    const std::string &GetValue() { return m_value; }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    std::string m_value;
  };

  class Dictionary : public Object {
  public:
    Dictionary() : Object(Type::eTypeDictionary), m_dict() {}

    ~Dictionary() override = default;

    size_t GetSize() const { return m_dict.size(); }

    void ForEach(std::function<bool(ConstString key, Object *object)> const
                     &callback) const {
      for (const auto &pair : m_dict) {
        if (callback(pair.first, pair.second.get()) == false)
          break;
      }
    }

    ObjectSP GetKeys() const {
      ObjectSP object_sp(new Array());
      Array *array = object_sp->GetAsArray();
      collection::const_iterator iter;
      for (iter = m_dict.begin(); iter != m_dict.end(); ++iter) {
        ObjectSP key_object_sp(new String());
        key_object_sp->GetAsString()->SetValue(iter->first.AsCString());
        array->Push(key_object_sp);
      }
      return object_sp;
    }

    ObjectSP GetValueForKey(llvm::StringRef key) const {
      ObjectSP value_sp;
      if (!key.empty()) {
        ConstString key_cs(key);
        collection::const_iterator iter = m_dict.find(key_cs);
        if (iter != m_dict.end())
          value_sp = iter->second;
      }
      return value_sp;
    }

    template <class IntType>
    bool GetValueForKeyAsInteger(llvm::StringRef key, IntType &result) const {
      ObjectSP value_sp = GetValueForKey(key);
      if (value_sp) {
        if (auto int_value = value_sp->GetAsInteger()) {
          result = static_cast<IntType>(int_value->GetValue());
          return true;
        }
      }
      return false;
    }

    template <class IntType>
    bool GetValueForKeyAsInteger(llvm::StringRef key, IntType &result,
                                 IntType default_val) const {
      bool success = GetValueForKeyAsInteger<IntType>(key, result);
      if (!success)
        result = default_val;
      return success;
    }

    bool GetValueForKeyAsString(llvm::StringRef key,
                                std::string &result) const {
      ObjectSP value_sp = GetValueForKey(key);
      if (value_sp.get()) {
        if (auto string_value = value_sp->GetAsString()) {
          result = string_value->GetValue();
          return true;
        }
      }
      return false;
    }

    bool GetValueForKeyAsString(llvm::StringRef key, std::string &result,
                                const char *default_val) const {
      bool success = GetValueForKeyAsString(key, result);
      if (!success) {
        if (default_val)
          result = default_val;
        else
          result.clear();
      }
      return success;
    }

    bool GetValueForKeyAsString(llvm::StringRef key,
                                ConstString &result) const {
      ObjectSP value_sp = GetValueForKey(key);
      if (value_sp.get()) {
        if (auto string_value = value_sp->GetAsString()) {
          result = ConstString(string_value->GetValue());
          return true;
        }
      }
      return false;
    }

    bool GetValueForKeyAsString(llvm::StringRef key, ConstString &result,
                                const char *default_val) const {
      bool success = GetValueForKeyAsString(key, result);
      if (!success)
        result.SetCString(default_val);
      return success;
    }

    bool GetValueForKeyAsDictionary(llvm::StringRef key,
                                    Dictionary *&result) const {
      result = nullptr;
      ObjectSP value_sp = GetValueForKey(key);
      if (value_sp.get()) {
        result = value_sp->GetAsDictionary();
        return (result != nullptr);
      }
      return false;
    }

    bool GetValueForKeyAsArray(llvm::StringRef key, Array *&result) const {
      result = nullptr;
      ObjectSP value_sp = GetValueForKey(key);
      if (value_sp.get()) {
        result = value_sp->GetAsArray();
        return (result != nullptr);
      }
      return false;
    }

    bool HasKey(llvm::StringRef key) const {
      ConstString key_cs(key);
      collection::const_iterator search = m_dict.find(key_cs);
      return search != m_dict.end();
    }

    void AddItem(llvm::StringRef key, ObjectSP value_sp) {
      ConstString key_cs(key);
      m_dict[key_cs] = value_sp;
    }

    void AddIntegerItem(llvm::StringRef key, uint64_t value) {
      AddItem(key, ObjectSP(new Integer(value)));
    }

    void AddFloatItem(llvm::StringRef key, double value) {
      AddItem(key, ObjectSP(new Float(value)));
    }

    void AddStringItem(llvm::StringRef key, std::string value) {
      AddItem(key, ObjectSP(new String(std::move(value))));
    }

    void AddBooleanItem(llvm::StringRef key, bool value) {
      AddItem(key, ObjectSP(new Boolean(value)));
    }

    void Dump(Stream &s, bool pretty_print = true) const override;

  protected:
    typedef std::map<ConstString, ObjectSP> collection;
    collection m_dict;
  };

  class Null : public Object {
  public:
    Null() : Object(Type::eTypeNull) {}

    ~Null() override = default;

    bool IsValid() const override { return false; }

    void Dump(Stream &s, bool pretty_print = true) const override;
  };

  class Generic : public Object {
  public:
    explicit Generic(void *object = nullptr)
        : Object(Type::eTypeGeneric), m_object(object) {}

    void SetValue(void *value) { m_object = value; }

    void *GetValue() const { return m_object; }

    bool IsValid() const override { return m_object != nullptr; }

    void Dump(Stream &s, bool pretty_print = true) const override;

  private:
    void *m_object;
  };

  static ObjectSP ParseJSON(std::string json_text);
};

} // namespace lldb_private

#endif // liblldb_StructuredData_h_
