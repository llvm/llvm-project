/*  cilk.h                  -*-C++-*-
 *
 *  Copyright (C) 2010-2018, Intel Corporation
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *    * Neither the name of Intel Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  *********************************************************************
 *  
 *  PLEASE NOTE: This file is a downstream copy of a file maintained in
 *  a repository at cilkplus.org. Changes made to this file that are not
 *  submitted through the contribution process detailed at
 *  http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 *  time that a new version is released. Changes only submitted to the
 *  GNU compiler collection or posted to the git repository at
 *  https://bitbucket.org/intelcilkruntime/intel-cilk-runtime are
 *  not tracked.
 *  
 *  We welcome your contributions to this open source project. Thank you
 *  for your assistance in helping us improve Cilk Plus.
 */
 
/** @file cilk.h
 *
 *  @brief Provides convenient aliases for Intel(R) Cilk(TM) language keywords.
 *
 *  @details
 *  Since Intel Cilk Plus is a nonstandard extension to both C and C++, the Intel
 *  Cilk language keywords all begin with "`_Cilk_`", which guarantees that they
 *  will not conflict with user-defined identifiers in properly written 
 *  programs. This way, a Cilk-enabled C or C++ compiler can safely compile 
 *  "standard" C and C++ programs.
 *
 *  However, this means that the keywords _look_ like something grafted on to
 *  the base language. Therefore, you can include this header:
 *
 *      #include "cilk/cilk.h"
 *
 *  and then write the Intel Cilk keywords with a "`cilk_`" prefix instead of
 *  "`_Cilk_`".
 *
 *  @ingroup language
 */
 
 
/** @defgroup language Language Keywords
 *  Definitions for the Intel Cilk language.
 *  @{
 */
 
#ifndef cilk_spawn
# define cilk_spawn _Cilk_spawn ///< Spawn a task that can execute in parallel.
# define cilk_sync  _Cilk_sync  ///< Wait for spawned tasks to complete.
# define cilk_for   _Cilk_for   ///< Execute iterations of a `for` loop in parallel.
#endif

/// @}
