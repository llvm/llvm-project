#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv09  ########


kv09: run
	

build:  $(SRC)/kv09.f
	-$(RM) kv09.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv09.f -o kv09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv09.$(OBJX) check.$(OBJX) $(LIBS) -o kv09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv09
	kv09.$(EXESUFFIX)

verify: ;

kv09.run: run

