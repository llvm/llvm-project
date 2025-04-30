# Copyright (C) 1995-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

#
#	Sub-makefile for sysvipc portion of the library.
#
subdir	:= sysvipc

include ../Makeconfig

headers	:= sys/ipc.h sys/msg.h sys/sem.h sys/shm.h \
	   bits/ipctypes.h bits/ipc.h bits/msq.h bits/sem.h bits/shm.h

routines := ftok \
	    msgsnd msgrcv msgget msgctl \
	    semop semget semctl semtimedop \
	    shmat shmdt shmget shmctl

tests    := test-sysvmsg test-sysvsem test-sysvshm

tests-time64 := test-sysvmsg-time64 test-sysvsem-time64 test-sysvshm-time64

include ../Rules

CFLAGS-msgrcv.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-msgsnd.c += -fexceptions -fasynchronous-unwind-tables

$(objpfx)test-sysvsem: $(librt)
$(objpfx)test-sysvsem-time64: $(librt)
