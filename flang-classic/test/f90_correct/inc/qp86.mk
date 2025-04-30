#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtoint8 ########


qp86: run


build:  $(SRC)/qp86.f08
	-$(RM) qp86.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp86.f08 -o qp86.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp86.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qp86.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp86
	qp86.$(EXESUFFIX)

verify: ;


