#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb10  ########


hb10: run
	

build:  $(SRC)/hb10.f
	-$(RM) hb10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb10.f -o hb10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb10.$(OBJX) check.$(OBJX) $(LIBS) -o hb10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb10
	hb10.$(EXESUFFIX)

verify: ;

hb10.run: run

