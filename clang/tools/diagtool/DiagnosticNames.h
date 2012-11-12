//===- DiagnosticNames.h - Defines a table of all builtin diagnostics ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace diagtool {

  struct DiagnosticRecord {
    const char *NameStr;
    short DiagID;
    uint8_t NameLen;
    
    llvm::StringRef getName() const {
      return llvm::StringRef(NameStr, NameLen);
    }

    bool operator<(const DiagnosticRecord &Other) const {
      return getName() < Other.getName();
    }
  };

  /// \brief Get every diagnostic in the system, sorted by name.
  llvm::ArrayRef<DiagnosticRecord> getBuiltinDiagnosticsByName();

  /// \brief Get a diagnostic by its ID.
  const DiagnosticRecord &getDiagnosticForID(short DiagID);


  struct GroupRecord {
    // Be safe with the size of 'NameLen' because we don't statically check if
    // the size will fit in the field; the struct size won't decrease with a
    // shorter type anyway.
    size_t NameLen;
    const char *NameStr;
    const short *Members;
    const short *SubGroups;
    
    llvm::StringRef getName() const {
      return llvm::StringRef(NameStr, NameLen);
    }

    template<typename RecordType>
    class group_iterator {
      const short *CurrentID;

      friend struct GroupRecord;
      group_iterator(const short *Start) : CurrentID(Start) {
        if (CurrentID && *CurrentID == -1)
          CurrentID = 0;
      }

    public:
      typedef RecordType                 value_type;
      typedef const value_type &         reference;
      typedef const value_type *         pointer;
      typedef std::forward_iterator_tag  iterator_category;
      typedef std::ptrdiff_t             difference_type;

      inline reference operator*() const;
      inline pointer operator->() const {
        return &operator*();
      }

      inline short getID() const {
        return *CurrentID;
      }

      group_iterator &operator++() {
        ++CurrentID;
        if (*CurrentID == -1)
          CurrentID = 0;
        return *this;
      }

      bool operator==(group_iterator &Other) const {
        return CurrentID == Other.CurrentID;
      }
      
      bool operator!=(group_iterator &Other) const {
        return CurrentID != Other.CurrentID;
      }
    };

    typedef group_iterator<GroupRecord> subgroup_iterator;
    subgroup_iterator subgroup_begin() const {
      return SubGroups;
    }
    subgroup_iterator subgroup_end() const {
      return 0;
    }

    typedef group_iterator<DiagnosticRecord> diagnostics_iterator;
    diagnostics_iterator diagnostics_begin() const {
      return Members;
    }
    diagnostics_iterator diagnostics_end() const {
      return 0;
    }

    bool operator<(const GroupRecord &Other) const {
      return getName() < Other.getName();
    }
  };

  /// \brief Get every diagnostic group in the system, sorted by name.
  llvm::ArrayRef<GroupRecord> getDiagnosticGroups();

  template<>
  inline GroupRecord::subgroup_iterator::reference
  GroupRecord::subgroup_iterator::operator*() const {
    return getDiagnosticGroups()[*CurrentID];
  }

  template<>
  inline GroupRecord::diagnostics_iterator::reference
  GroupRecord::diagnostics_iterator::operator*() const {
    return getDiagnosticForID(*CurrentID);
  }
} // end namespace diagtool

