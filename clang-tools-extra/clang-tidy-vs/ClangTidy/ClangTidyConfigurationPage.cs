using Microsoft.VisualStudio;
using Microsoft.VisualStudio.Shell;
using Microsoft.VisualStudio.Shell.Interop;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LLVM.ClangTidy
{
    [ClassInterface(ClassInterfaceType.AutoDual)]
    [CLSCompliant(false), ComVisible(true)]
    public class ClangTidyConfigurationPage : DialogPage
    {
        ClangTidyPropertyGrid Grid = null;
        protected override IWin32Window Window
        {
            get
            {
                if (Grid == null)
                    Grid = new ClangTidyPropertyGrid();
                return Grid;
            }
        }

        protected override void SaveSetting(PropertyDescriptor property)
        {
            base.SaveSetting(property);
        }

        public override void SaveSettingsToStorage()
        {
            if (Grid != null)
                Grid.SaveSettingsToStorage();

            base.SaveSettingsToStorage();
        }

        public override void ResetSettings()
        {
            base.ResetSettings();
        }

        protected override void LoadSettingFromStorage(PropertyDescriptor prop)
        {
            base.LoadSettingFromStorage(prop);
        }

        public override void LoadSettingsFromStorage()
        {
            if (Grid != null)
                Grid.InitializeSettings();
            base.LoadSettingsFromStorage();
        }
    }
}
