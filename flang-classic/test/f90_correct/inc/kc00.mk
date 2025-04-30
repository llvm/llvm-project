#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kc00  ########


kc00: run
	

build:  $(SRC)/kc00.f
	-$(RM) kc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kc00.f -o kc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kc00.$(OBJX) check.$(OBJX) $(LIBS) -o kc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kc00
	kc00.$(EXESUFFIX)

verify: ;

kc00.run: run

