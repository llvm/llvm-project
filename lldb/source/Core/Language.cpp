//===-- Language.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Core/Language.h"
#include "lldb/Core/Stream.h"
#include <string.h>

using namespace lldb;
using namespace lldb_private;

#define ENUM_TO_DCSTREAM(x) case x: s->PutCString(#x); return

struct LanguageStrings
{
    const char * names[3];
};

static LanguageStrings
g_languages[] =
{
    { { "unknown" , NULL          , NULL                  } },
    { { "c89"     , NULL          , "ISO C:1989"          } },
    { { NULL      , NULL          , "K&R C"               } },
    { { "ada83"   , "Ada83"       , "ISO Ada:1983"        } },
    { { "c++"     , "cxx"         , "ISO C++:1998"        } },
    { { "cobol74" , "Cobol74"     , "ISO Cobol:1974"      } },
    { { "cobol"   , "Cobol85"     , "ISO Cobol:1985."     } },
    { { "f77"     , "Fortran77"   , "ISO Fortran 77."     } },
    { { "f90"     , "Fortran90"   , "ISO Fortran 90"      } },
    { { "pascal"  , "Pascal83"    , "ISO Pascal:1983"     } },
    { { "modula2" , "Modula2"     , "ISO Modula-2:1996"   } },
    { { "java"    , NULL          , "Java"                } },
    { { "c"       , "C99"         , "ISO C:1999"          } },
    { { "ada"     , "Ada95"       , "ISO Ada:1995"        } },
    { { "f95"     , "Fortran95"   , "ISO Fortran 95"      } },
    { { "PLI"     , NULL          , "ANSI PL/I:1976"      } },
    { { "objc"    , NULL          , "Objective-C"         } },
    { { "objc++"  , NULL          , "Objective-C++"       } },
    { { "upc"     , NULL          , "Unified Parallel C"  } },
    { { "d"       , NULL          , "D"                   } },
    { { "python"  , NULL          , "Python"              } }
};

static const size_t
g_num_languages = sizeof(g_languages)/sizeof(LanguageStrings);

Language::Language(LanguageType language) :
    m_language (language)
{
}

Language::~Language()
{
}

LanguageType
Language::GetLanguage() const
{
    return m_language;
}

void
Language::Clear ()
{
    m_language = eLanguageTypeUnknown;
}

void
Language::SetLanguage(LanguageType language)
{
    m_language = language;
}

bool
Language::SetLanguageFromCString(const char *language_cstr)
{
    size_t i, desc_idx;
    const char *name;

    // First check the most common name for the languages
    for (desc_idx=lldb::eDescriptionLevelBrief; desc_idx<kNumDescriptionLevels; ++desc_idx)
    {
        for (i=0; i<g_num_languages; ++i)
        {
            name = g_languages[i].names[desc_idx];
            if (name == NULL)
                continue;

            if (::strcasecmp (language_cstr, name) == 0)
            {
                m_language = (LanguageType)i;
                return true;
            }
        }
    }

    m_language = eLanguageTypeUnknown;
    return false;
}


const char *
Language::AsCString (lldb::DescriptionLevel level) const
{
    if (m_language < g_num_languages && level < kNumDescriptionLevels)
    {
        const char *name = g_languages[m_language].names[level];
        if (name)
            return name;
        else if (level + 1 < kNumDescriptionLevels)
            return AsCString ((lldb::DescriptionLevel)(level + 1));
        else
            return NULL;
    }
    return NULL;
}

void
Language::Dump(Stream *s) const
{
    GetDescription(s, lldb::eDescriptionLevelVerbose);
}

void
Language::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    const char *lang_cstr = AsCString(level);

    if (lang_cstr)
        s->PutCString(lang_cstr);
    else
        s->Printf("Language(language = 0x%4.4x)", m_language);
}




Stream&
lldb_private::operator << (Stream& s, const Language& language)
{
    language.Dump(&s);
    return s;
}

