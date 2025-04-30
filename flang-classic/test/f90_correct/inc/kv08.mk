#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv08  ########


kv08: run
	

build:  $(SRC)/kv08.f
	-$(RM) kv08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv08.f -o kv08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv08.$(OBJX) check.$(OBJX) $(LIBS) -o kv08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv08
	kv08.$(EXESUFFIX)

verify: ;

kv08.run: run

