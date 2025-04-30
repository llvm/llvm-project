#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv01  ########


kv01: run
	

build:  $(SRC)/kv01.f
	-$(RM) kv01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv01.f -o kv01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv01.$(OBJX) check.$(OBJX) $(LIBS) -o kv01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv01
	kv01.$(EXESUFFIX)

verify: ;

kv01.run: run

