#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par10: par10.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par10.$(OBJX): $(SRC)/par10.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par10.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par10.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par10
run: ;
