#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb40  ########


hb40: run
	

build:  $(SRC)/hb40.f
	-$(RM) hb40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb40.f -o hb40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb40.$(OBJX) check.$(OBJX) $(LIBS) -o hb40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb40
	hb40.$(EXESUFFIX)

verify: ;

hb40.run: run

