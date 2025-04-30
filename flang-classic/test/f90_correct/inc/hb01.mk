#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb01  ########


hb01: run
	

build:  $(SRC)/hb01.f
	-$(RM) hb01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb01.f -o hb01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb01.$(OBJX) check.$(OBJX) $(LIBS) -o hb01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb01
	hb01.$(EXESUFFIX)

verify: ;

hb01.run: run

