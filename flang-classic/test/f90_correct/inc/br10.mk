#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test br10  ########


br10: run
	

build:  $(SRC)/br10.f
	-$(RM) br10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/br10.f -o br10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) br10.$(OBJX) check.$(OBJX) $(LIBS) -o br10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test br10
	br10.$(EXESUFFIX)

verify: ;

br10.run: run

