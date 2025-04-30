#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv07  ########


kv07: run
	

build:  $(SRC)/kv07.f
	-$(RM) kv07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv07.f -o kv07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv07.$(OBJX) check.$(OBJX) $(LIBS) -o kv07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv07
	kv07.$(EXESUFFIX)

verify: ;

kv07.run: run

