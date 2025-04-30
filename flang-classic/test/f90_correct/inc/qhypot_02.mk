#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qhypot_02  ########


qhypot_02: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qhypot_02.f08 $(SRC)/qhypot_02_expct.c fcheck.$(OBJX)
	-$(RM) qhypot_02.$(EXESUFFIX) qhypot_02.$(OBJX) qhypot_02_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/qhypot_02_expct.c -o qhypot_02_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qhypot_02.f08 -o qhypot_02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qhypot_02.$(OBJX) qhypot_02_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qhypot_02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qhypot_02
	qhypot_02.$(EXESUFFIX)

verify: ;

qhypot_02.run: run

