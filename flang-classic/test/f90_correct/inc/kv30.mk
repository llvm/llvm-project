#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv30  ########


kv30: run
	

build:  $(SRC)/kv30.f
	-$(RM) kv30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv30.f -o kv30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv30.$(OBJX) check.$(OBJX) $(LIBS) -o kv30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv30
	kv30.$(EXESUFFIX)

verify: ;

kv30.run: run

