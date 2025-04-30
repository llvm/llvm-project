#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv02  ########


kv02: run
	

build:  $(SRC)/kv02.f
	-$(RM) kv02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv02.f -o kv02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv02.$(OBJX) check.$(OBJX) $(LIBS) -o kv02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv02
	kv02.$(EXESUFFIX)

verify: ;

kv02.run: run

