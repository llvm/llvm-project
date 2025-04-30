#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv05  ########


kv05: run
	

build:  $(SRC)/kv05.f
	-$(RM) kv05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv05.f -o kv05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv05.$(OBJX) check.$(OBJX) $(LIBS) -o kv05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv05
	kv05.$(EXESUFFIX)

verify: ;

kv05.run: run

