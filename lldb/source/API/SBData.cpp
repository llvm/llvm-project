//===-- SBData.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBData.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Stream.h"

#include <cinttypes>
#include <memory>

using namespace lldb;
using namespace lldb_private;

SBData::SBData() : m_opaque_sp(new DataExtractor()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBData);
}

SBData::SBData(const lldb::DataExtractorSP &data_sp) : m_opaque_sp(data_sp) {}

SBData::SBData(const SBData &rhs) : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBData, (const lldb::SBData &), rhs);
}

const SBData &SBData::operator=(const SBData &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBData &,
                     SBData, operator=,(const lldb::SBData &), rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

SBData::~SBData() {}

void SBData::SetOpaque(const lldb::DataExtractorSP &data_sp) {
  m_opaque_sp = data_sp;
}

lldb_private::DataExtractor *SBData::get() const { return m_opaque_sp.get(); }

lldb_private::DataExtractor *SBData::operator->() const {
  return m_opaque_sp.operator->();
}

lldb::DataExtractorSP &SBData::operator*() { return m_opaque_sp; }

const lldb::DataExtractorSP &SBData::operator*() const { return m_opaque_sp; }

bool SBData::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBData, IsValid);

  return m_opaque_sp.get() != NULL;
}

uint8_t SBData::GetAddressByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint8_t, SBData, GetAddressByteSize);

  uint8_t value = 0;
  if (m_opaque_sp.get())
    value = m_opaque_sp->GetAddressByteSize();
  return value;
}

void SBData::SetAddressByteSize(uint8_t addr_byte_size) {
  LLDB_RECORD_METHOD(void, SBData, SetAddressByteSize, (uint8_t),
                     addr_byte_size);

  if (m_opaque_sp.get())
    m_opaque_sp->SetAddressByteSize(addr_byte_size);
}

void SBData::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBData, Clear);

  if (m_opaque_sp.get())
    m_opaque_sp->Clear();
}

size_t SBData::GetByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBData, GetByteSize);

  size_t value = 0;
  if (m_opaque_sp.get())
    value = m_opaque_sp->GetByteSize();
  return value;
}

lldb::ByteOrder SBData::GetByteOrder() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ByteOrder, SBData, GetByteOrder);

  lldb::ByteOrder value = eByteOrderInvalid;
  if (m_opaque_sp.get())
    value = m_opaque_sp->GetByteOrder();
  return value;
}

void SBData::SetByteOrder(lldb::ByteOrder endian) {
  LLDB_RECORD_METHOD(void, SBData, SetByteOrder, (lldb::ByteOrder), endian);

  if (m_opaque_sp.get())
    m_opaque_sp->SetByteOrder(endian);
}

float SBData::GetFloat(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(float, SBData, GetFloat, (lldb::SBError &, lldb::offset_t),
                     error, offset);

  float value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetFloat(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

double SBData::GetDouble(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(double, SBData, GetDouble,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  double value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetDouble(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

long double SBData::GetLongDouble(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(long double, SBData, GetLongDouble,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  long double value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetLongDouble(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

lldb::addr_t SBData::GetAddress(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(lldb::addr_t, SBData, GetAddress,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  lldb::addr_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetAddress(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

uint8_t SBData::GetUnsignedInt8(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(uint8_t, SBData, GetUnsignedInt8,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  uint8_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetU8(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

uint16_t SBData::GetUnsignedInt16(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(uint16_t, SBData, GetUnsignedInt16,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  uint16_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetU16(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

uint32_t SBData::GetUnsignedInt32(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(uint32_t, SBData, GetUnsignedInt32,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  uint32_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetU32(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

uint64_t SBData::GetUnsignedInt64(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(uint64_t, SBData, GetUnsignedInt64,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  uint64_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetU64(&offset);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

int8_t SBData::GetSignedInt8(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(int8_t, SBData, GetSignedInt8,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  int8_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = (int8_t)m_opaque_sp->GetMaxS64(&offset, 1);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

int16_t SBData::GetSignedInt16(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(int16_t, SBData, GetSignedInt16,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  int16_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = (int16_t)m_opaque_sp->GetMaxS64(&offset, 2);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

int32_t SBData::GetSignedInt32(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(int32_t, SBData, GetSignedInt32,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  int32_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = (int32_t)m_opaque_sp->GetMaxS64(&offset, 4);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

int64_t SBData::GetSignedInt64(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(int64_t, SBData, GetSignedInt64,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  int64_t value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = (int64_t)m_opaque_sp->GetMaxS64(&offset, 8);
    if (offset == old_offset)
      error.SetErrorString("unable to read data");
  }
  return value;
}

const char *SBData::GetString(lldb::SBError &error, lldb::offset_t offset) {
  LLDB_RECORD_METHOD(const char *, SBData, GetString,
                     (lldb::SBError &, lldb::offset_t), error, offset);

  const char *value = 0;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    value = m_opaque_sp->GetCStr(&offset);
    if (offset == old_offset || (value == NULL))
      error.SetErrorString("unable to read data");
  }
  return value;
}

bool SBData::GetDescription(lldb::SBStream &description,
                            lldb::addr_t base_addr) {
  LLDB_RECORD_METHOD(bool, SBData, GetDescription,
                     (lldb::SBStream &, lldb::addr_t), description, base_addr);

  Stream &strm = description.ref();

  if (m_opaque_sp) {
    DumpDataExtractor(*m_opaque_sp, &strm, 0, lldb::eFormatBytesWithASCII, 1,
                      m_opaque_sp->GetByteSize(), 16, base_addr, 0, 0);
  } else
    strm.PutCString("No value");

  return true;
}

size_t SBData::ReadRawData(lldb::SBError &error, lldb::offset_t offset,
                           void *buf, size_t size) {
  void *ok = NULL;
  if (!m_opaque_sp.get()) {
    error.SetErrorString("no value to read from");
  } else {
    uint32_t old_offset = offset;
    ok = m_opaque_sp->GetU8(&offset, buf, size);
    if ((offset == old_offset) || (ok == NULL))
      error.SetErrorString("unable to read data");
  }
  return ok ? size : 0;
}

void SBData::SetData(lldb::SBError &error, const void *buf, size_t size,
                     lldb::ByteOrder endian, uint8_t addr_size) {
  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buf, size, endian, addr_size);
  else
  {
    m_opaque_sp->SetData(buf, size, endian);
    m_opaque_sp->SetAddressByteSize(addr_size);
  }

}

bool SBData::Append(const SBData &rhs) {
  LLDB_RECORD_METHOD(bool, SBData, Append, (const lldb::SBData &), rhs);

  bool value = false;
  if (m_opaque_sp.get() && rhs.m_opaque_sp.get())
    value = m_opaque_sp.get()->Append(*rhs.m_opaque_sp);
  return value;
}

lldb::SBData SBData::CreateDataFromCString(lldb::ByteOrder endian,
                                           uint32_t addr_byte_size,
                                           const char *data) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromCString,
                            (lldb::ByteOrder, uint32_t, const char *), endian,
                            addr_byte_size, data);

  if (!data || !data[0])
    return LLDB_RECORD_RESULT(SBData());

  uint32_t data_len = strlen(data);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(data, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

lldb::SBData SBData::CreateDataFromUInt64Array(lldb::ByteOrder endian,
                                               uint32_t addr_byte_size,
                                               uint64_t *array,
                                               size_t array_len) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromUInt64Array,
                            (lldb::ByteOrder, uint32_t, uint64_t *, size_t),
                            endian, addr_byte_size, array, array_len);

  if (!array || array_len == 0)
    return LLDB_RECORD_RESULT(SBData());

  size_t data_len = array_len * sizeof(uint64_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

lldb::SBData SBData::CreateDataFromUInt32Array(lldb::ByteOrder endian,
                                               uint32_t addr_byte_size,
                                               uint32_t *array,
                                               size_t array_len) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromUInt32Array,
                            (lldb::ByteOrder, uint32_t, uint32_t *, size_t),
                            endian, addr_byte_size, array, array_len);

  if (!array || array_len == 0)
    return LLDB_RECORD_RESULT(SBData());

  size_t data_len = array_len * sizeof(uint32_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

lldb::SBData SBData::CreateDataFromSInt64Array(lldb::ByteOrder endian,
                                               uint32_t addr_byte_size,
                                               int64_t *array,
                                               size_t array_len) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromSInt64Array,
                            (lldb::ByteOrder, uint32_t, int64_t *, size_t),
                            endian, addr_byte_size, array, array_len);

  if (!array || array_len == 0)
    return LLDB_RECORD_RESULT(SBData());

  size_t data_len = array_len * sizeof(int64_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

lldb::SBData SBData::CreateDataFromSInt32Array(lldb::ByteOrder endian,
                                               uint32_t addr_byte_size,
                                               int32_t *array,
                                               size_t array_len) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromSInt32Array,
                            (lldb::ByteOrder, uint32_t, int32_t *, size_t),
                            endian, addr_byte_size, array, array_len);

  if (!array || array_len == 0)
    return LLDB_RECORD_RESULT(SBData());

  size_t data_len = array_len * sizeof(int32_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

lldb::SBData SBData::CreateDataFromDoubleArray(lldb::ByteOrder endian,
                                               uint32_t addr_byte_size,
                                               double *array,
                                               size_t array_len) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromDoubleArray,
                            (lldb::ByteOrder, uint32_t, double *, size_t),
                            endian, addr_byte_size, array, array_len);

  if (!array || array_len == 0)
    return LLDB_RECORD_RESULT(SBData());

  size_t data_len = array_len * sizeof(double);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));
  lldb::DataExtractorSP data_sp(
      new DataExtractor(buffer_sp, endian, addr_byte_size));

  SBData ret(data_sp);

  return LLDB_RECORD_RESULT(ret);
}

bool SBData::SetDataFromCString(const char *data) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromCString, (const char *), data);


  if (!data) {
    return false;
  }

  size_t data_len = strlen(data);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(data, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);


  return true;
}

bool SBData::SetDataFromUInt64Array(uint64_t *array, size_t array_len) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromUInt64Array, (uint64_t *, size_t),
                     array, array_len);


  if (!array || array_len == 0) {
    return false;
  }

  size_t data_len = array_len * sizeof(uint64_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);


  return true;
}

bool SBData::SetDataFromUInt32Array(uint32_t *array, size_t array_len) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromUInt32Array, (uint32_t *, size_t),
                     array, array_len);


  if (!array || array_len == 0) {
    return false;
  }

  size_t data_len = array_len * sizeof(uint32_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);

  return true;
}

bool SBData::SetDataFromSInt64Array(int64_t *array, size_t array_len) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromSInt64Array, (int64_t *, size_t),
                     array, array_len);


  if (!array || array_len == 0) {
    return false;
  }

  size_t data_len = array_len * sizeof(int64_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);

  return true;
}

bool SBData::SetDataFromSInt32Array(int32_t *array, size_t array_len) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromSInt32Array, (int32_t *, size_t),
                     array, array_len);


  if (!array || array_len == 0) {
    return false;
  }

  size_t data_len = array_len * sizeof(int32_t);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);

  return true;
}

bool SBData::SetDataFromDoubleArray(double *array, size_t array_len) {
  LLDB_RECORD_METHOD(bool, SBData, SetDataFromDoubleArray, (double *, size_t),
                     array, array_len);


  if (!array || array_len == 0) {
    return false;
  }

  size_t data_len = array_len * sizeof(double);

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(array, data_len));

  if (!m_opaque_sp.get())
    m_opaque_sp = std::make_shared<DataExtractor>(buffer_sp, GetByteOrder(),
                                                  GetAddressByteSize());
  else
    m_opaque_sp->SetData(buffer_sp);

  return true;
}
