#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtobint  ########


qp80: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp80.f08 fcheck.$(OBJX)
	-$(RM) qp80.$(EXESUFFIX) qp80.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp80.f08 -o qp80.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp80.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp80.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp80
	qp80.$(EXESUFFIX)

verify: ;

qp80.run: run

