#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv13  ########


kv13: run
	

build:  $(SRC)/kv13.f
	-$(RM) kv13.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv13.f -o kv13.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv13.$(OBJX) check.$(OBJX) $(LIBS) -o kv13.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv13
	kv13.$(EXESUFFIX)

verify: ;

kv13.run: run

