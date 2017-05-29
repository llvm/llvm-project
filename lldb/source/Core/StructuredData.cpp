//===---------------------StructuredData.cpp ---------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StructuredData.h"

#include "lldb/Host/File.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/JSON.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h" // for Stream
#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h" // for FilePermissions::eFilePermiss...
#include "lldb/lldb-forward.h"      // for DataBufferSP

#include "llvm/ADT/STLExtras.h" // for make_unique

#include <limits> // for numeric_limits

#include <errno.h>
#include <inttypes.h>
#include <stdio.h> // for printf
#include <stdlib.h>
#include <sys/types.h> // for off_t

using namespace lldb_private;

//----------------------------------------------------------------------
// Functions that use a JSONParser to parse JSON into StructuredData
//----------------------------------------------------------------------
static StructuredData::ObjectSP ParseJSONValue(JSONParser &json_parser);
static StructuredData::ObjectSP ParseJSONObject(JSONParser &json_parser);
static StructuredData::ObjectSP ParseJSONArray(JSONParser &json_parser);

StructuredData::ObjectSP
StructuredData::ParseJSONFromFile(const FileSpec &input_spec, Status &error) {
  StructuredData::ObjectSP return_sp;
  if (!input_spec.Exists()) {
    error.SetErrorStringWithFormat("input file %s does not exist.",
                                   input_spec.GetPath().c_str());
    return return_sp;
  }

  File input_file(nullptr, File::OpenOptions::eOpenOptionRead,
                  lldb::eFilePermissionsUserRead);
  std::string input_path = input_spec.GetPath();
  error =
      input_file.Open(input_path.c_str(), File::OpenOptions::eOpenOptionRead,
                      lldb::eFilePermissionsUserRead);

  if (!error.Success()) {
    error.SetErrorStringWithFormat("could not open input file: %s - %s.",
                                   input_spec.GetPath().c_str(),
                                   error.AsCString());
    return return_sp;
  }

  lldb::DataBufferSP input_data;
  size_t num_bytes = std::numeric_limits<size_t>::max();
  off_t offset = 0;
  error = input_file.Read(num_bytes, offset, true, input_data);
  if (!error.Success()) {
    error.SetErrorStringWithFormat("could not read input file: %s - %s.",
                                   input_spec.GetPath().c_str(),
                                   error.AsCString());
    return return_sp;
  }
  JSONParser json_parser((char *)input_data->GetBytes());
  return_sp = ParseJSONValue(json_parser);
  return return_sp;
}

static StructuredData::ObjectSP ParseJSONObject(JSONParser &json_parser) {
  // The "JSONParser::Token::ObjectStart" token should have already been
  // consumed by the time this function is called
  auto dict_up = llvm::make_unique<StructuredData::Dictionary>();

  std::string value;
  std::string key;
  while (1) {
    JSONParser::Token token = json_parser.GetToken(value);

    if (token == JSONParser::Token::String) {
      key.swap(value);
      token = json_parser.GetToken(value);
      if (token == JSONParser::Token::Colon) {
        StructuredData::ObjectSP value_sp = ParseJSONValue(json_parser);
        if (value_sp)
          dict_up->AddItem(key, value_sp);
        else
          break;
      }
    } else if (token == JSONParser::Token::ObjectEnd) {
      return StructuredData::ObjectSP(dict_up.release());
    } else if (token == JSONParser::Token::Comma) {
      continue;
    } else {
      break;
    }
  }
  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONArray(JSONParser &json_parser) {
  // The "JSONParser::Token::ObjectStart" token should have already been
  // consumed
  // by the time this function is called
  auto array_up = llvm::make_unique<StructuredData::Array>();

  std::string value;
  std::string key;
  while (1) {
    StructuredData::ObjectSP value_sp = ParseJSONValue(json_parser);
    if (value_sp)
      array_up->AddItem(value_sp);
    else
      break;

    JSONParser::Token token = json_parser.GetToken(value);
    if (token == JSONParser::Token::Comma) {
      continue;
    } else if (token == JSONParser::Token::ArrayEnd) {
      return StructuredData::ObjectSP(array_up.release());
    } else {
      break;
    }
  }
  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONValue(JSONParser &json_parser) {
  std::string value;
  const JSONParser::Token token = json_parser.GetToken(value);
  switch (token) {
  case JSONParser::Token::ObjectStart:
    return ParseJSONObject(json_parser);

  case JSONParser::Token::ArrayStart:
    return ParseJSONArray(json_parser);

  case JSONParser::Token::Integer: {
    bool success = false;
    uint64_t uval = StringConvert::ToUInt64(value.c_str(), 0, 0, &success);
    if (success)
      return std::make_shared<StructuredData::Integer>(uval);
  } break;

  case JSONParser::Token::Float: {
    bool success = false;
    double val = StringConvert::ToDouble(value.c_str(), 0.0, &success);
    if (success)
      return std::make_shared<StructuredData::Float>(val);
  } break;

  case JSONParser::Token::String:
    return std::make_shared<StructuredData::String>(value);

  case JSONParser::Token::True:
  case JSONParser::Token::False:
    return std::make_shared<StructuredData::Boolean>(token ==
                                                     JSONParser::Token::True);

  case JSONParser::Token::Null:
    return std::make_shared<StructuredData::Null>();

  default:
    break;
  }
  return StructuredData::ObjectSP();
}

StructuredData::ObjectSP StructuredData::ParseJSON(std::string json_text) {
  JSONParser json_parser(json_text.c_str());
  StructuredData::ObjectSP object_sp = ParseJSONValue(json_parser);
  return object_sp;
}

StructuredData::ObjectSP
StructuredData::Object::GetObjectForDotSeparatedPath(llvm::StringRef path) {
  if (this->GetType() == lldb::eStructuredDataTypeDictionary) {
    std::pair<llvm::StringRef, llvm::StringRef> match = path.split('.');
    std::string key = match.first.str();
    ObjectSP value = this->GetAsDictionary()->GetValueForKey(key);
    if (value.get()) {
      // Do we have additional words to descend?  If not, return the
      // value we're at right now.
      if (match.second.empty()) {
        return value;
      } else {
        return value->GetObjectForDotSeparatedPath(match.second);
      }
    }
    return ObjectSP();
  }

  if (this->GetType() == lldb::eStructuredDataTypeArray) {
    std::pair<llvm::StringRef, llvm::StringRef> match = path.split('[');
    if (match.second.size() == 0) {
      return this->shared_from_this();
    }
    errno = 0;
    uint64_t val = strtoul(match.second.str().c_str(), NULL, 10);
    if (errno == 0) {
      return this->GetAsArray()->GetItemAtIndex(val);
    }
    return ObjectSP();
  }

  return this->shared_from_this();
}

void StructuredData::Object::DumpToStdout(bool pretty_print) const {
  StreamString stream;
  Dump(stream, pretty_print);
  printf("%s\n", stream.GetData());
}

void StructuredData::Array::Dump(Stream &s, bool pretty_print) const {
  bool first = true;
  s << "[";
  if (pretty_print) {
    s << "\n";
    s.IndentMore();
  }
  for (const auto &item_sp : m_items) {
    if (first) {
      first = false;
    } else {
      s << ",";
      if (pretty_print)
        s << "\n";
    }

    if (pretty_print)
      s.Indent();
    item_sp->Dump(s, pretty_print);
  }
  if (pretty_print) {
    s.IndentLess();
    s.EOL();
    s.Indent();
  }
  s << "]";
}

void StructuredData::Integer::Dump(Stream &s, bool pretty_print) const {
  s.Printf("%" PRIu64, m_value);
}

void StructuredData::Float::Dump(Stream &s, bool pretty_print) const {
  s.Printf("%lg", m_value);
}

void StructuredData::Boolean::Dump(Stream &s, bool pretty_print) const {
  if (m_value == true)
    s.PutCString("true");
  else
    s.PutCString("false");
}

void StructuredData::String::Dump(Stream &s, bool pretty_print) const {
  std::string quoted;
  const size_t strsize = m_value.size();
  for (size_t i = 0; i < strsize; ++i) {
    char ch = m_value[i];
    if (ch == '"' || ch == '\\')
      quoted.push_back('\\');
    quoted.push_back(ch);
  }
  s.Printf("\"%s\"", quoted.c_str());
}

void StructuredData::Dictionary::Dump(Stream &s, bool pretty_print) const {
  bool first = true;
  s << "{";
  if (pretty_print) {
    s << "\n";
    s.IndentMore();
  }
  for (const auto &pair : m_dict) {
    if (first)
      first = false;
    else {
      s << ",";
      if (pretty_print)
        s << "\n";
    }
    if (pretty_print)
      s.Indent();
    s << "\"" << pair.first.AsCString() << "\" : ";
    pair.second->Dump(s, pretty_print);
  }
  if (pretty_print) {
    s.IndentLess();
    s.EOL();
    s.Indent();
  }
  s << "}";
}

void StructuredData::Null::Dump(Stream &s, bool pretty_print) const {
  s << "null";
}

void StructuredData::Generic::Dump(Stream &s, bool pretty_print) const {
  s << "0x" << m_object;
}
