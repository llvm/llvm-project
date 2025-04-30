#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka01  ########


ka01: run
	

build:  $(SRC)/ka01.f
	-$(RM) ka01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka01.f -o ka01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka01.$(OBJX) check.$(OBJX) $(LIBS) -o ka01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka01
	ka01.$(EXESUFFIX)

verify: ;

ka01.run: run

