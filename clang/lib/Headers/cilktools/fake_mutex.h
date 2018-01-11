/* fake_mutex.h                  -*-C++-*-
 *
 *************************************************************************
 *
 * Copyright (C) 2013-2016, Intel Corporation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * *********************************************************************
 * 
 * PLEASE NOTE: This file is a downstream copy of a file mainitained in
 * a repository at cilkplus.org. Changes made to this file that are not
 * submitted through the contribution process detailed at
 * http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 * time that a new version is released. Changes only submitted to the
 * GNU compiler collection or posted to the git repository at
 * https://bitbucket.org/intelcilkruntime/itnel-cilk-runtime.git are
 * not tracked.
 * 
 * We welcome your contributions to this open source project. Thank you
 * for your assistance in helping us improve Cilk Plus.
 *
 **************************************************************************
 *
 * Cilkscreen fake mutexes are provided to indicate to the Cilkscreen race
 * detector that a race should be ignored.
 *
 * NOTE: This class does not provide mutual exclusion.  You should use the
 * mutual exclusion constructs provided by TBB or your operating system to
 * protect against real data races.
 */

#ifndef FAKE_MUTEX_H_INCLUDED
#define FAKE_MUTEX_H_INCLUDED

#include <cilktools/cilkscreen.h>

namespace cilkscreen
{
    class fake_mutex
    {
    public:
	fake_mutex() : locked(false)
	{
	}

	~fake_mutex()
	{
	    __CILKRTS_ASSERT(! locked);
	}

        // Wait until mutex is available, then enter
        void lock()
        {
            __cilkscreen_acquire_lock(&locked);
	    __CILKRTS_ASSERT(! locked);
	    locked = true;
        }

        // A fake mutex is always available
        bool try_lock() { lock(); return true; }

        // Releases the mutex
        void unlock()
        {
	    __CILKRTS_ASSERT(locked);
	    locked = false;
            __cilkscreen_release_lock(&locked);
        }

    private:
        bool locked;
    };

} // namespace cilk

#endif  // FAKE_MUTEX_H_INCLUDED
