#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qhypot_01  ########


qhypot_01: run
	

build:  $(SRC)/qhypot_01.f08
	-$(RM) qhypot_01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qhypot_01.f08 -o qhypot_01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qhypot_01.$(OBJX) check.$(OBJX) $(LIBS) -o qhypot_01.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qhypot_01 
	qhypot_01.$(EXESUFFIX)

verify: ;


