//===-- ClangTidyPackages.cs - VSPackage for clang-tidy ----------*- C# -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains a VS extension package that runs clang-tidy over a
// file in a VS text editor.
//
//===----------------------------------------------------------------------===//

using Microsoft.VisualStudio.Editor;
using Microsoft.VisualStudio.Shell;
using Microsoft.VisualStudio.Shell.Interop;
using Microsoft.VisualStudio.TextManager.Interop;
using System;
using System.Collections;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Xml.Linq;

namespace LLVM.ClangTidy
{
    [PackageRegistration(UseManagedResourcesOnly = true)]
    [InstalledProductRegistration("#110", "#112", "1.0", IconResourceID = 400)]
    [ProvideMenuResource("Menus.ctmenu", 1)]
    [Guid(GuidList.guidClangTidyPkgString)]
    [ProvideOptionPage(typeof(ClangTidyConfigurationPage), "LLVM/Clang", "ClangTidy", 0, 0, true)]
    public sealed class ClangTidyPackage : Package
    {
        #region Package Members
        protected override void Initialize()
        {
            base.Initialize();

            var commandService = GetService(typeof(IMenuCommandService)) as OleMenuCommandService;
            if (commandService != null)
            {
                var menuCommandID = new CommandID(GuidList.guidClangTidyCmdSet, (int)PkgCmdIDList.cmdidClangTidy);
                var menuItem = new MenuCommand(MenuItemCallback, menuCommandID);
                commandService.AddCommand(menuItem);
            }
        }
        #endregion

        private void MenuItemCallback(object sender, EventArgs args)
        {
        }
    }
}
