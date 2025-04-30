# Pretty printers for the NPTL lock types.
#
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

"""This file contains the gdb pretty printers for the following types:

    * pthread_mutex_t
    * pthread_mutexattr_t
    * pthread_cond_t
    * pthread_condattr_t
    * pthread_rwlock_t
    * pthread_rwlockattr_t

You can check which printers are registered and enabled by issuing the
'info pretty-printer' gdb command.  Printers should trigger automatically when
trying to print a variable of one of the types mentioned above.
"""

from __future__ import print_function

import gdb
import gdb.printing
from nptl_lock_constants import *

MUTEX_TYPES = {
    PTHREAD_MUTEX_NORMAL: ('Type', 'Normal'),
    PTHREAD_MUTEX_RECURSIVE: ('Type', 'Recursive'),
    PTHREAD_MUTEX_ERRORCHECK: ('Type', 'Error check'),
    PTHREAD_MUTEX_ADAPTIVE_NP: ('Type', 'Adaptive')
}

class MutexPrinter(object):
    """Pretty printer for pthread_mutex_t."""

    def __init__(self, mutex):
        """Initialize the printer's internal data structures.

        Args:
            mutex: A gdb.value representing a pthread_mutex_t.
        """

        data = mutex['__data']
        self.lock = data['__lock']
        self.count = data['__count']
        self.owner = data['__owner']
        self.kind = data['__kind']
        self.values = []
        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutex_t.
        """

        return 'pthread_mutex_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutex_t.
        """

        return self.values

    def read_values(self):
        """Read the mutex's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_type()
        self.read_status()
        self.read_attributes()
        self.read_misc_info()

    def read_type(self):
        """Read the mutex's type."""

        mutex_type = self.kind & PTHREAD_MUTEX_KIND_MASK

        # mutex_type must be casted to int because it's a gdb.Value
        self.values.append(MUTEX_TYPES[int(mutex_type)])

    def read_status(self):
        """Read the mutex's status.

        Architectures that support lock elision might not record the mutex owner
        ID in the __owner field.  In that case, the owner will be reported as
        "Unknown".
        """

        if self.kind == PTHREAD_MUTEX_DESTROYED:
            self.values.append(('Status', 'Destroyed'))
        elif self.kind & PTHREAD_MUTEX_ROBUST_NORMAL_NP:
            self.read_status_robust()
        else:
            self.read_status_no_robust()

    def read_status_robust(self):
        """Read the status of a robust mutex.

        In glibc robust mutexes are implemented in a very different way than
        non-robust ones.  This method reads their locking status,
        whether it may have waiters, their registered owner (if any),
        whether the owner is alive or not, and the status of the state
        they're protecting.
        """

        if self.lock == PTHREAD_MUTEX_UNLOCKED:
            self.values.append(('Status', 'Not acquired'))
        else:
            if self.lock & FUTEX_WAITERS:
                self.values.append(('Status',
                                    'Acquired, possibly with waiters'))
            else:
                self.values.append(('Status',
                                    'Acquired, possibly with no waiters'))

            if self.lock & FUTEX_OWNER_DIED:
                self.values.append(('Owner ID', '%d (dead)' % self.owner))
            else:
                self.values.append(('Owner ID', self.lock & FUTEX_TID_MASK))

        if self.owner == PTHREAD_MUTEX_INCONSISTENT:
            self.values.append(('State protected by this mutex',
                                'Inconsistent'))
        elif self.owner == PTHREAD_MUTEX_NOTRECOVERABLE:
            self.values.append(('State protected by this mutex',
                                'Not recoverable'))

    def read_status_no_robust(self):
        """Read the status of a non-robust mutex.

        Read info on whether the mutex is acquired, if it may have waiters
        and its owner (if any).
        """

        lock_value = self.lock

        if self.kind & PTHREAD_MUTEX_PRIO_PROTECT_NP:
            lock_value &= 0xffffffff & ~(PTHREAD_MUTEX_PRIO_CEILING_MASK)

        if lock_value == PTHREAD_MUTEX_UNLOCKED:
            self.values.append(('Status', 'Not acquired'))
        else:
            if self.kind & PTHREAD_MUTEX_PRIO_INHERIT_NP:
                waiters = self.lock & FUTEX_WAITERS
                owner = self.lock & FUTEX_TID_MASK
            else:
                # Mutex protocol is PP or none
                waiters = (self.lock != PTHREAD_MUTEX_LOCKED_NO_WAITERS)
                owner = self.owner

            if waiters:
                self.values.append(('Status',
                                    'Acquired, possibly with waiters'))
            else:
                self.values.append(('Status',
                                    'Acquired, possibly with no waiters'))

            if self.owner != 0:
                self.values.append(('Owner ID', owner))
            else:
                # Owner isn't recorded, probably because lock elision
                # is enabled.
                self.values.append(('Owner ID', 'Unknown'))

    def read_attributes(self):
        """Read the mutex's attributes."""

        if self.kind != PTHREAD_MUTEX_DESTROYED:
            if self.kind & PTHREAD_MUTEX_ROBUST_NORMAL_NP:
                self.values.append(('Robust', 'Yes'))
            else:
                self.values.append(('Robust', 'No'))

            # In glibc, robust mutexes always have their pshared flag set to
            # 'shared' regardless of what the pshared flag of their
            # mutexattr was.  Therefore a robust mutex will act as shared
            # even if it was initialized with a 'private' mutexattr.
            if self.kind & PTHREAD_MUTEX_PSHARED_BIT:
                self.values.append(('Shared', 'Yes'))
            else:
                self.values.append(('Shared', 'No'))

            if self.kind & PTHREAD_MUTEX_PRIO_INHERIT_NP:
                self.values.append(('Protocol', 'Priority inherit'))
            elif self.kind & PTHREAD_MUTEX_PRIO_PROTECT_NP:
                prio_ceiling = ((self.lock & PTHREAD_MUTEX_PRIO_CEILING_MASK)
                                >> PTHREAD_MUTEX_PRIO_CEILING_SHIFT)

                self.values.append(('Protocol', 'Priority protect'))
                self.values.append(('Priority ceiling', prio_ceiling))
            else:
                # PTHREAD_PRIO_NONE
                self.values.append(('Protocol', 'None'))

    def read_misc_info(self):
        """Read miscellaneous info on the mutex.

        For now this reads the number of times a recursive mutex was acquired
        by the same thread.
        """

        mutex_type = self.kind & PTHREAD_MUTEX_KIND_MASK

        if mutex_type == PTHREAD_MUTEX_RECURSIVE and self.count > 1:
            self.values.append(('Times acquired by the owner', self.count))

class MutexAttributesPrinter(object):
    """Pretty printer for pthread_mutexattr_t.

    In the NPTL this is a type that's always casted to struct pthread_mutexattr
    which has a single 'mutexkind' field containing the actual attributes.
    """

    def __init__(self, mutexattr):
        """Initialize the printer's internal data structures.

        Args:
            mutexattr: A gdb.value representing a pthread_mutexattr_t.
        """

        self.values = []

        try:
            mutexattr_struct = gdb.lookup_type('struct pthread_mutexattr')
            self.mutexattr = mutexattr.cast(mutexattr_struct)['mutexkind']
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', mutexattr['__size']))
            self.values.append(('__align', mutexattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutexattr_t.
        """

        return 'pthread_mutexattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutexattr_t.
        """

        return self.values

    def read_values(self):
        """Read the mutexattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        mutexattr_type = (self.mutexattr
                          & 0xffffffff
                          & ~PTHREAD_MUTEXATTR_FLAG_BITS
                          & ~PTHREAD_MUTEX_NO_ELISION_NP)

        # mutexattr_type must be casted to int because it's a gdb.Value
        self.values.append(MUTEX_TYPES[int(mutexattr_type)])

        if self.mutexattr & PTHREAD_MUTEXATTR_FLAG_ROBUST:
            self.values.append(('Robust', 'Yes'))
        else:
            self.values.append(('Robust', 'No'))

        if self.mutexattr & PTHREAD_MUTEXATTR_FLAG_PSHARED:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

        protocol = ((self.mutexattr & PTHREAD_MUTEXATTR_PROTOCOL_MASK) >>
                    PTHREAD_MUTEXATTR_PROTOCOL_SHIFT)

        if protocol == PTHREAD_PRIO_NONE:
            self.values.append(('Protocol', 'None'))
        elif protocol == PTHREAD_PRIO_INHERIT:
            self.values.append(('Protocol', 'Priority inherit'))
        elif protocol == PTHREAD_PRIO_PROTECT:
            self.values.append(('Protocol', 'Priority protect'))

class ConditionVariablePrinter(object):
    """Pretty printer for pthread_cond_t."""

    def __init__(self, cond):
        """Initialize the printer's internal data structures.

        Args:
            cond: A gdb.value representing a pthread_cond_t.
        """

        data = cond['__data']
        self.wrefs = data['__wrefs']
        self.values = []

        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_cond_t.
        """

        return 'pthread_cond_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_cond_t.
        """

        return self.values

    def read_values(self):
        """Read the condvar's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_status()
        self.read_attributes()

    def read_status(self):
        """Read the status of the condvar.

        This method reads whether the condvar is destroyed and how many threads
        are waiting for it.
        """

        self.values.append(('Threads known to still execute a wait function',
                            self.wrefs >> PTHREAD_COND_WREFS_SHIFT))

    def read_attributes(self):
        """Read the condvar's attributes."""

        if (self.wrefs & PTHREAD_COND_CLOCK_MONOTONIC_MASK) != 0:
            self.values.append(('Clock ID', 'CLOCK_MONOTONIC'))
        else:
            self.values.append(('Clock ID', 'CLOCK_REALTIME'))

        if (self.wrefs & PTHREAD_COND_SHARED_MASK) != 0:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

class ConditionVariableAttributesPrinter(object):
    """Pretty printer for pthread_condattr_t.

    In the NPTL this is a type that's always casted to struct pthread_condattr,
    which has a single 'value' field containing the actual attributes.
    """

    def __init__(self, condattr):
        """Initialize the printer's internal data structures.

        Args:
            condattr: A gdb.value representing a pthread_condattr_t.
        """

        self.values = []

        try:
            condattr_struct = gdb.lookup_type('struct pthread_condattr')
            self.condattr = condattr.cast(condattr_struct)['value']
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', condattr['__size']))
            self.values.append(('__align', condattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_condattr_t.
        """

        return 'pthread_condattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_condattr_t.
        """

        return self.values

    def read_values(self):
        """Read the condattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        clock_id = (self.condattr >> 1) & ((1 << COND_CLOCK_BITS) - 1)

        if clock_id != 0:
            self.values.append(('Clock ID', 'CLOCK_MONOTONIC'))
        else:
            self.values.append(('Clock ID', 'CLOCK_REALTIME'))

        if self.condattr & 1:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

class RWLockPrinter(object):
    """Pretty printer for pthread_rwlock_t."""

    def __init__(self, rwlock):
        """Initialize the printer's internal data structures.

        Args:
            rwlock: A gdb.value representing a pthread_rwlock_t.
        """

        data = rwlock['__data']
        self.readers = data['__readers']
        self.cur_writer = data['__cur_writer']
        self.shared = data['__shared']
        self.flags = data['__flags']
        self.values = []
        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlock_t.
        """

        return 'pthread_rwlock_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlock_t.
        """

        return self.values

    def read_values(self):
        """Read the rwlock's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_status()
        self.read_attributes()

    def read_status(self):
        """Read the status of the rwlock."""

        if self.readers & PTHREAD_RWLOCK_WRPHASE:
            if self.readers & PTHREAD_RWLOCK_WRLOCKED:
                self.values.append(('Status', 'Acquired (Write)'))
                self.values.append(('Writer ID', self.cur_writer))
            else:
                self.values.append(('Status', 'Not acquired'))
        else:
            r = self.readers >> PTHREAD_RWLOCK_READER_SHIFT
            if r > 0:
                self.values.append(('Status', 'Acquired (Read)'))
                self.values.append(('Readers', r))
            else:
                self.values.append(('Status', 'Not acquired'))

    def read_attributes(self):
        """Read the attributes of the rwlock."""

        if self.shared:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

        if self.flags == PTHREAD_RWLOCK_PREFER_READER_NP:
            self.values.append(('Prefers', 'Readers'))
        elif self.flags == PTHREAD_RWLOCK_PREFER_WRITER_NP:
            self.values.append(('Prefers', 'Writers'))
        else:
            self.values.append(('Prefers', 'Writers no recursive readers'))

class RWLockAttributesPrinter(object):
    """Pretty printer for pthread_rwlockattr_t.

    In the NPTL this is a type that's always casted to
    struct pthread_rwlockattr, which has two fields ('lockkind' and 'pshared')
    containing the actual attributes.
    """

    def __init__(self, rwlockattr):
        """Initialize the printer's internal data structures.

        Args:
            rwlockattr: A gdb.value representing a pthread_rwlockattr_t.
        """

        self.values = []

        try:
            rwlockattr_struct = gdb.lookup_type('struct pthread_rwlockattr')
            self.rwlockattr = rwlockattr.cast(rwlockattr_struct)
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', rwlockattr['__size']))
            self.values.append(('__align', rwlockattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlockattr_t.
        """

        return 'pthread_rwlockattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlockattr_t.
        """

        return self.values

    def read_values(self):
        """Read the rwlockattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        rwlock_type = self.rwlockattr['lockkind']
        shared = self.rwlockattr['pshared']

        if shared == PTHREAD_PROCESS_SHARED:
            self.values.append(('Shared', 'Yes'))
        else:
            # PTHREAD_PROCESS_PRIVATE
            self.values.append(('Shared', 'No'))

        if rwlock_type == PTHREAD_RWLOCK_PREFER_READER_NP:
            self.values.append(('Prefers', 'Readers'))
        elif rwlock_type == PTHREAD_RWLOCK_PREFER_WRITER_NP:
            self.values.append(('Prefers', 'Writers'))
        else:
            self.values.append(('Prefers', 'Writers no recursive readers'))

def register(objfile):
    """Register the pretty printers within the given objfile."""

    printer = gdb.printing.RegexpCollectionPrettyPrinter('glibc-pthread-locks')

    printer.add_printer('pthread_mutex_t', r'^pthread_mutex_t$',
                        MutexPrinter)
    printer.add_printer('pthread_mutexattr_t', r'^pthread_mutexattr_t$',
                        MutexAttributesPrinter)
    printer.add_printer('pthread_cond_t', r'^pthread_cond_t$',
                        ConditionVariablePrinter)
    printer.add_printer('pthread_condattr_t', r'^pthread_condattr_t$',
                        ConditionVariableAttributesPrinter)
    printer.add_printer('pthread_rwlock_t', r'^pthread_rwlock_t$',
                        RWLockPrinter)
    printer.add_printer('pthread_rwlockattr_t', r'^pthread_rwlockattr_t$',
                        RWLockAttributesPrinter)

    if objfile == None:
        objfile = gdb

    gdb.printing.register_pretty_printer(objfile, printer)

register(gdb.current_objfile())
