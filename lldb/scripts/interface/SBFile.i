//===-- SWIG Interface for SBFile -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a file."
) SBFile;

class SBFile
{
public:
    SBFile();
    SBFile(int fd, const char *mode, bool transfer_ownership);

    ~SBFile ();

    %feature("autodoc", "Read(buffer) -> SBError, bytes_read") Read;
    SBError Read(uint8_t *buf, size_t num_bytes, size_t *OUTPUT);

    %feature("autodoc", "Write(buffer) -> SBError, written_read") Write;
    SBError Write(const uint8_t *buf, size_t num_bytes, size_t *OUTPUT);

    void Flush();

    bool IsValid() const;

    operator bool() const;

    SBError Close();
};

} // namespace lldb
