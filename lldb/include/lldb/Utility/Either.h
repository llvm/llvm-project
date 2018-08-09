//===-- Either.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Either_h_
#define liblldb_Either_h_

#include "llvm/ADT/Optional.h"

namespace lldb_utility {
    template <typename T1, typename T2>
    class Either {
    private:
        enum class Selected {
            One, Two
        };
        
        Selected m_selected;
        union {
            T1 m_t1;
            T2 m_t2;
        };
        
    public:
        Either(const T1& t1)
        {
            m_t1 = t1;
            m_selected = Selected::One;
        }
        
        Either(const T2& t2)
        {
            m_t2 = t2;
            m_selected = Selected::Two;
        }
        
        template <class X, typename std::enable_if<std::is_same<T1,X>::value>::type * = nullptr >
        llvm::Optional<T1>
        GetAs()
        {
            switch (m_selected)
            {
                case Selected::One:
                    return m_t1;
                default:
                    return llvm::Optional<T1>();
            }
        }
        
        template <class X, typename std::enable_if<std::is_same<T2,X>::value>::type * = nullptr >
        llvm::Optional<T2>
        GetAs()
        {
            switch (m_selected)
            {
                case Selected::Two:
                    return m_t2;
                default:
                    return llvm::Optional<T2>();
            }
        }
    };
    
} // namespace lldb_utility

#endif // #ifndef liblldb_Either_h_

