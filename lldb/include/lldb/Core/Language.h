//===-- Language.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Language_h_
#define liblldb_Language_h_

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Language Language.h "lldb/Core/Language.h"
/// @brief Encapsulates the programming language for an lldb object.
///
/// Languages are represented by an enumeration value.
///
/// The enumeration values used when describing the programming language
/// are the same values as the latest DWARF specification.
//----------------------------------------------------------------------
class Language
{
public:
    //------------------------------------------------------------------
    /// Construct with optional language enumeration.
    //------------------------------------------------------------------
    Language(lldb::LanguageType language = lldb::eLanguageTypeUnknown);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~Language();

    //------------------------------------------------------------------
    /// Get the language value as a NULL termianted C string.
    ///
    /// @return
    ///     The C string representation of the language. The returned
    ///     string does not need to be freed as it comes from constant
    ///     strings. NULL can be returned when the language is set to
    ///     a value that doesn't match of of the lldb::LanguageType
    ///     enumerations.
    //------------------------------------------------------------------
    const char *
    AsCString (lldb::DescriptionLevel level = lldb::eDescriptionLevelBrief) const;

    void
    Clear();

    void
    GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    //------------------------------------------------------------------
    /// Dump the language value to the stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump the language description.
    //------------------------------------------------------------------
    void
    Dump(Stream *s) const;

    //------------------------------------------------------------------
    /// Get accessor for the language.
    ///
    /// @return
    ///     The enumeration value that describes the programming
    ///     language that an object is associated with.
    //------------------------------------------------------------------
    virtual lldb::LanguageType
    GetLanguage() const;

    //------------------------------------------------------------------
    /// Set accessor for the language.
    ///
    /// @param[in] language
    ///     The new enumeration value that describes the programming
    ///     language that an object is associated with.
    //------------------------------------------------------------------
    void
    SetLanguage(lldb::LanguageType language);

    //------------------------------------------------------------------
    /// Set accessor for the language.
    ///
    /// @param[in] language_cstr
    ///     The language name as a C string.
    //------------------------------------------------------------------
    bool
    SetLanguageFromCString(const char *language_cstr);


protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    lldb::LanguageType m_language; ///< The programming language enumeration value.
                                   ///< The enumeration values are the same as the
                                   ///< latest DWARF specification.
};

//--------------------------------------------------------------
/// Stream the language enumeration as a string object to a
/// Stream.
//--------------------------------------------------------------
Stream& operator << (Stream& s, const Language& language);

} // namespace lldb_private

#endif  // liblldb_Language_h_
